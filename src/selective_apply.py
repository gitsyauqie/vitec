#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eksperimen MURAH (tanpa retrain): terapkan koreksi GEC secara SELEKTIF.
Untuk tiap kalimat: pakai output GEC hanya bila confidence >= tau (dan/atau
perubahan thd source tidak terlalu besar); selain itu pertahankan source.

Tujuan: naikkan presisi edit → hentikan over-correction yg bikin CER +1.5pp.

Input: CSV dari gen_eval_da.py (kolom: source, hyp, kalimat_koreksi, gec_score, category).
Output: tabel CER vs tau + guard, plus baseline & batas-atas ORACLE (headroom).

Pakai:
  python selective_apply.py --pred results/e2e_C1.csv --cer-norm lower
  python selective_apply.py --pred results/e2e_C2.csv --cer-norm lower
"""
import argparse, csv, re, statistics

# ── CER (samakan dgn eval_protocol_v2: default lower, tanda baca dipertahankan) ──
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
        cur = [i] + [0]*lb
        ai = a[i-1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ai==b[j-1] else 1))
        prev = cur
    return prev[lb]

def corpus_cer(pairs, mode):
    """pairs: list (hyp, ref). micro CER."""
    e = r = 0
    for hyp, ref in pairs:
        rn = norm(ref, mode); hn = norm(hyp, mode)
        e += lev(list(rn), list(hn)); r += len(rn)
    return e / r if r else 0.0

def rel_editdist(a, b, mode):
    an = norm(a, mode); bn = norm(b, mode)
    if not an: return 0.0
    return lev(list(an), list(bn)) / len(an)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--cer-norm", default="lower", choices=["raw","lower","lower_nopunct"])
    ap.add_argument("--max-change", type=float, default=None,
                    help="opsional: tolak hyp bila rel-editdist(source,hyp) > nilai ini")
    args = ap.parse_args()
    M = args.cer_norm

    rows = list(csv.DictReader(open(args.pred, encoding="utf-8")))
    src = [r["source"] for r in rows]
    hyp = [r["hyp"] for r in rows]
    ref = [r["kalimat_koreksi"] for r in rows]
    score = [float(r["gec_score"]) if r.get("gec_score") not in (None,"") else -99.0 for r in rows]
    n = len(rows)

    base_cer = corpus_cer(list(zip(src, ref)), M)            # tanpa GEC
    full_cer = corpus_cer(list(zip(hyp, ref)), M)            # semua GEC
    # oracle selektif: per kalimat pilih yg CER-nya lebih kecil (batas atas gating)
    oracle = []
    for s, h, rf in zip(src, hyp, ref):
        oracle.append(h if corpus_cer([(h,rf)],M) <= corpus_cer([(s,rf)],M) else s)
    oracle_cer = corpus_cer(list(zip(oracle, ref)), M)
    pct_hyp_better = 100*sum(1 for s,h,rf in zip(src,hyp,ref)
                             if corpus_cer([(h,rf)],M) < corpus_cer([(s,rf)],M))/n

    print(f"File: {args.pred}  (N={n}, cer-norm={M})")
    print(f"  baseline (source, no-GEC) : {base_cer*100:.2f}%")
    print(f"  full GEC (semua diterapkan): {full_cer*100:.2f}%")
    print(f"  ORACLE selektif (batas atas): {oracle_cer*100:.2f}%   [%kalimat GEC>source: {pct_hyp_better:.1f}%]")
    print(f"  → headroom gating: {base_cer*100:.2f}% (no-GEC) ↔ {oracle_cer*100:.2f}% (oracle)")
    print()

    # sweep tau pada persentil skor
    qs = [0,10,20,30,40,50,60,70,80,90,100]
    taus = [statistics.quantiles(score, n=100)[min(q,99)-1] if 0<q<100 else (min(score) if q==0 else max(score)) for q in qs]
    print(f"  {'tau-pct':>7} {'tau':>8} {'%corrected':>11} {'CER':>8}")
    best = (base_cer, "no-GEC")
    for q, tau in zip(qs, taus):
        out = []
        nc = 0
        for s, h, sc in zip(src, hyp, score):
            take = sc >= tau
            if take and args.max_change is not None:
                if rel_editdist(s, h, M) > args.max_change:
                    take = False
            out.append(h if take else s)
            nc += int(take)
        cer = corpus_cer(list(zip(out, ref)), M)
        print(f"  {q:>6}% {tau:>8.3f} {100*nc/n:>10.1f}% {cer*100:>7.2f}%")
        if cer < best[0]:
            best = (cer, f"tau-pct={q} ({100*nc/n:.0f}% corrected)")
    print(f"\n  BEST selektif: {best[0]*100:.2f}%  @ {best[1]}")
    print(f"  (target kalahkan baseline no-GEC {base_cer*100:.2f}%)")

if __name__ == "__main__":
    main()
