#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Terapkan ambang gating TETAP (dipilih di split validasi) ke satu set prediksi.
TIDAK ada sweep di sini -- itu justru intinya: angka yang dilaporkan di paper
harus berasal dari tau yang dikunci sebelum melihat test.

Pakai:
  python apply_fixed_tau.py --pred results/e2e_C3_nm2.csv --tau -0.2360 \
      --cer-norm lower_nopunct

Keluaran: CER tanpa GEC, CER dengan gating pada tau, cakupan (%), dan
sign-test berpasangan gated-vs-source. Tulis --out untuk menyimpan hipotesis
akhir per kalimat (siap dipakai eval_protocol_v2.py / paired_test.py).
"""
import argparse, csv, math, re

def norm(s, mode):
    if not s: return ""
    if mode == "raw": return s
    s = s.lower()
    if mode == "lower_nopunct": s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def lev(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[lb]

def corpus_cer(pairs, mode):
    e = r = 0
    for hyp, ref in pairs:
        rn, hn = norm(ref, mode), norm(hyp, mode)
        e += lev(list(rn), list(hn)); r += len(rn)
    return e / r if r else 0.0

def sent_cer(hyp, ref, mode):
    rn, hn = norm(ref, mode), norm(hyp, mode)
    return lev(list(rn), list(hn)) / len(rn) if rn else 0.0

def sign_test(better, worse):
    """dua sisi, p-value eksak binomial(0.5) pada pasangan yang berbeda."""
    n = better + worse
    if n == 0: return 1.0
    k = min(better, worse)
    c = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * c / (2 ** n))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tau", type=float, required=True,
                    help="ambang yang DIPILIH DI VALIDASI -- jangan disetel di sini")
    ap.add_argument("--cer-norm", default="lower_nopunct",
                    choices=["raw", "lower", "lower_nopunct"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    M = args.cer_norm

    rows = list(csv.DictReader(open(args.pred, encoding="utf-8")))
    out, n_corr, better, worse = [], 0, 0, 0
    for r in rows:
        s, h, ref = r["source"], r["hyp"], r["kalimat_koreksi"]
        sc = float(r["gec_score"]) if r.get("gec_score") not in (None, "") else -99.0
        take = sc > args.tau
        y = h if take else s
        n_corr += int(take)
        if take:
            ds, dh = sent_cer(s, ref, M), sent_cer(h, ref, M)
            if dh < ds: better += 1
            elif dh > ds: worse += 1
        out.append((r.get("id", ""), r.get("category", ""), s, y, ref, sc, int(take)))

    n = len(rows)
    base = corpus_cer([(r["source"], r["kalimat_koreksi"]) for r in rows], M)
    gated = corpus_cer([(o[3], o[4]) for o in out], M)
    full = corpus_cer([(r["hyp"], r["kalimat_koreksi"]) for r in rows], M)

    print(f"File: {args.pred}  (N={n}, cer-norm={M}, tau={args.tau:.4f} [fixed])")
    print(f"  tanpa GEC (normaliser saja) : {base*100:.2f}%")
    print(f"  semua edit diterapkan       : {full*100:.2f}%")
    print(f"  gating pada tau tetap       : {gated*100:.2f}%   "
          f"({100*n_corr/n:.1f}% kalimat dikoreksi)")
    print(f"  di antara yang dikoreksi    : {better} membaik / {worse} memburuk  "
          f"(sign-test p={sign_test(better, worse):.3f})")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "category", "source", "hyp", "kalimat_koreksi",
                        "gec_score", "applied"])
            w.writerows(out)
        print(f"→ {args.out}")

if __name__ == "__main__":
    main()
