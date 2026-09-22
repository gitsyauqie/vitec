#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A1 — uji signifikansi BERPASANGAN antar konfigurasi pipeline (jawab kritik reviewer:
CI tumpang tindih ≠ tak signifikan; uji paired mengontrol kesulitan per-kalimat).

Metode per pasangan (A,B), CER per-kalimat (lower_nopunct):
  - mean diff (B−A) + paired bootstrap 95% CI + p (two-sided)
  - sign test (Wilcoxon-lite): #kalimat A<B vs B<A, binomial p
  - McNemar pada exact-match (CER==0)

Input (urut baris = sama, 201 kalimat):
  e2e_C3_nm2.csv : source(=L2-normalized), hyp(=GEC), gec_score, kalimat_koreksi
  e2e_ocr_outputs.csv : paddleocr_out (=OCR mentah), kalimat_koreksi   [utk C0]

Konfigurasi: C0(OCR) · L2(source) · FULL(hyp semua) · GATED(hyp bila score>=tau).
"""
import argparse, csv, re, random, math, statistics

def norm(s):
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def lev(a, b):
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0]*lb; ai = a[i-1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if ai==b[j-1] else 1))
        prev = cur
    return prev[lb]

def cer(hyp, ref):
    rn = norm(ref); hn = norm(hyp)
    return lev(list(rn), list(hn)) / max(len(rn), 1)

def paired_bootstrap(a, b, n=5000, seed=1):
    """a,b = per-sentence CER. Return mean diff (b-a), 95% CI, two-sided p."""
    n_s = len(a); rng = random.Random(seed)
    diffs = [b[i]-a[i] for i in range(n_s)]
    point = sum(diffs)/n_s
    boot = []
    for _ in range(n):
        s = sum(diffs[rng.randrange(n_s)] for _ in range(n_s))/n_s
        boot.append(s)
    boot.sort()
    lo = boot[int(0.025*n)]; hi = boot[int(0.975*n)-1]
    # p: fraction of bootstrap means on opposite side of 0
    frac_gt = sum(1 for x in boot if x > 0)/n
    p = 2*min(frac_gt, 1-frac_gt)
    return point, lo, hi, p

def sign_test(a, b):
    """#A<B (B worse) vs #B<A (B better); binomial two-sided p."""
    nb = sum(1 for i in range(len(a)) if b[i] < a[i] - 1e-9)   # B better
    nw = sum(1 for i in range(len(a)) if b[i] > a[i] + 1e-9)   # B worse
    n = nb + nw
    if n == 0: return nb, nw, 1.0
    k = min(nb, nw)
    # two-sided binomial p (p=0.5)
    p = 2*sum(math.comb(n, i) for i in range(0, k+1)) / (2**n)
    return nb, nw, min(p, 1.0)

def mcnemar(a, b):
    """exact-match flip: A correct & B not (b01) vs B correct & A not (b10)."""
    b01 = sum(1 for i in range(len(a)) if a[i] == 0 and b[i] != 0)
    b10 = sum(1 for i in range(len(a)) if a[i] != 0 and b[i] == 0)
    n = b01 + b10
    if n == 0: return b01, b10, 1.0
    k = min(b01, b10)
    p = 2*sum(math.comb(n, i) for i in range(0, k+1)) / (2**n)
    return b01, b10, min(p, 1.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", default="results/e2e_C3_nm2.csv")
    ap.add_argument("--ocr", default="results/e2e_ocr_outputs.csv")
    ap.add_argument("--tau-pct", type=int, default=60, help="operating point (% paling percaya diri yg di-gate)")
    ap.add_argument("--llm", default=None, help="CSV baseline LLM (kolom hyp), urut sama")
    args = ap.parse_args()

    c3 = list(csv.DictReader(open(args.c3, encoding="utf-8")))
    ocr = list(csv.DictReader(open(args.ocr, encoding="utf-8")))
    assert len(c3) == len(ocr), (len(c3), len(ocr))
    ref = [r["kalimat_koreksi"] for r in c3]
    scores = [float(r["gec_score"]) for r in c3]
    # tau dari persentil skor (gate kalimat paling pede)
    tau = statistics.quantiles(scores, n=100)[max(1, 100-args.tau_pct)-1]

    cer_ocr  = [cer(ocr[i]["paddleocr_out"], ref[i]) for i in range(len(c3))]
    cer_l2   = [cer(c3[i]["source"],          ref[i]) for i in range(len(c3))]
    cer_full = [cer(c3[i]["hyp"],             ref[i]) for i in range(len(c3))]
    cer_gate = [cer(c3[i]["hyp"] if scores[i] >= tau else c3[i]["source"], ref[i]) for i in range(len(c3))]

    def m(x): return 100*sum(x)/len(x)
    print(f"N={len(c3)} | mean CER%:  OCR={m(cer_ocr):.2f}  L2={m(cer_l2):.2f}  "
          f"FULL={m(cer_full):.2f}  GATED@{args.tau_pct}%={m(cer_gate):.2f}")
    print()
    def report(name, A, B):
        pt, lo, hi, p = paired_bootstrap(A, B)
        nb, nw, ps = sign_test(A, B)
        b01, b10, pm = mcnemar(A, B)
        print(f"[{name}]  ΔmeanCER(B−A)={pt*100:+.2f}pp  CI95=[{lo*100:+.2f},{hi*100:+.2f}]  p_boot={p:.3f}")
        print(f"        sign-test: B-better={nb}  B-worse={nw}  p={ps:.3f} | "
              f"McNemar EM: A-only={b01} B-only={b10} p={pm:.3f}")
    report("OCR -> L2        (apakah L2 menolong?)", cer_ocr, cer_l2)
    report("L2  -> FULL-GEC  (ungated, tuning-free)", cer_l2, cer_full)
    report("L2  -> GATED     (operating point)",      cer_l2, cer_gate)
    report("OCR -> GATED     (pipeline penuh)",       cer_ocr, cer_gate)
    if args.llm:
        llm = list(csv.DictReader(open(args.llm, encoding="utf-8")))
        assert len(llm) == len(c3), (len(llm), len(c3))
        cer_llm = [cer(llm[i]["hyp"], ref[i]) for i in range(len(c3))]
        print(f"\n  DeepSeek/LLM mean CER% = {m(cer_llm):.2f}")
        report("GATED -> LLM     (LLM lebih baik dari pipeline kita?)", cer_gate, cer_llm)
        report("OCR   -> LLM     (LLM menolong vs OCR?)",               cer_ocr,  cer_llm)
    print("\nCatatan: tau dipilih pada operating point (ideal: tune di val). "
          "L2->FULL adalah uji bebas-tuning.")

if __name__ == "__main__":
    main()
