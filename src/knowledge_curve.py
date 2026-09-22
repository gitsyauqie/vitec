#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge-resource analysis untuk normaliser deterministik (bahan section
"knowledge representation & acquisition" di kbs_new.tex).

Dua eksperimen (scope ALG, test set, harness eval sama dgn L2_normalizer --eval):
  (1) ABLATION per-komponen: lookup / leet / censor (semua kombinasi) + old-leet
      baseline → P/R/F1/F0.5 + bootstrap CI (resample kalimat).
  (2) KURVA nilai-pengetahuan: F0.5 vs fraksi knowledge resource, disampel acak
      multi-seed — (a) lookup dictionary, (b) validity lexicon.

Pakai (server, env fairseq):
  python knowledge_curve.py --selftest        # verifikasi logika, tanpa file server
  python knowledge_curve.py --run             # full → results/knowledge_curve.csv
Opsi: --fracs 0,25,50,75,100  --seeds 5  --boot 1000  --out PATH
"""
import argparse, csv, json, os, random, string, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from L2_normalizer import (LEET, AMBIG_LETTER, PUNCT, Normalizer, load_validity_dict,
                           augment_vocab_from_splits, build_lookup, old_leet, normcmp,
                           ALGO, DICTF, TEST, TRAIN, VAL, MINFREQ)

# ── normalizer dgn saklar komponen ───────────────────────────────────────────
class ComponentNormalizer(Normalizer):
    """use_lookup / use_leet / use_censor: matikan komponen tanpa mengubah logika lain.
       Menonaktifkan = karakter tsb tidak diekspansi (kandidat=dirinya sendiri) →
       decoding gagal validasi kamus → token dibiarkan (perilaku 'never guess' tetap)."""
    def __init__(self, vocab, lookup, use_lookup=True, use_leet=True, use_censor=True, **kw):
        super().__init__(vocab, lookup if use_lookup else {}, **kw)
        self.use_leet, self.use_censor = use_leet, use_censor

    def _candidates(self, core):
        import itertools
        opts = []
        for ch in core:
            if ch in LEET and self.use_leet:            opts.append(LEET[ch])
            elif ch == "*" and self.use_censor:         opts.append(list(string.ascii_lowercase))
            elif ch in AMBIG_LETTER and self.use_leet:  opts.append(AMBIG_LETTER[ch])
            else:                                       opts.append([ch])
        total = 1
        for o in opts: total *= len(o)
        if total > self.max_combos: return []
        return ["".join(p) for p in itertools.product(*opts)]

# ── eval per-kalimat (agar bisa bootstrap) ───────────────────────────────────
def eval_sentences(fn, data, scope=("ALG",)):
    """return list of (tp, fp, fn) per kalimat — logika identik L2_normalizer.evaluate()."""
    sents = []
    for f in data:
        for s in f.get("sentences", []):
            src = s.get("teks_ocr", "")
            gold, allerr = {}, set()
            for e in s.get("errors", []):
                sl = (e.get("token_salah") or "").strip()
                bn = (e.get("token_benar") or "").strip()
                if not sl: continue
                allerr.add(sl.strip(PUNCT).lower())
                if e.get("category") in scope and " " not in sl:
                    gold[sl.strip(PUNCT).lower()] = bn
            tp = fp = fn_ = 0
            for tok in src.split():
                core = tok.strip(PUNCT).lower()
                pred = fn(tok)
                changed = (pred.strip(PUNCT).lower() != core)
                if core in gold:
                    if normcmp(pred, gold[core]): tp += 1
                    else: fn_ += 1
                elif core not in allerr and changed:
                    fp += 1
            sents.append((tp, fp, fn_))
    return sents

def prf(tp, fp, fn_):
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn_) if (tp+fn_) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    f05 = 1.25*p*r/(0.25*p+r) if (0.25*p+r) else 0.0
    return p, r, f1, f05

def point_and_ci(sents, boot=1000, seed=0):
    tp = sum(x[0] for x in sents); fp = sum(x[1] for x in sents); fn_ = sum(x[2] for x in sents)
    P, R, F1, F05 = prf(tp, fp, fn_)
    lo = hi = F05
    if boot > 0 and sents:
        rng, n, vals = random.Random(seed), len(sents), []
        for _ in range(boot):
            t = f = g = 0
            for i in (rng.randrange(n) for _ in range(n)):
                t += sents[i][0]; f += sents[i][1]; g += sents[i][2]
            vals.append(prf(t, f, g)[3])
        vals.sort()
        lo, hi = vals[int(0.025*boot)], vals[min(int(0.975*boot), boot-1)]
    return dict(tp=tp, fp=fp, fn=fn_, P=P, R=R, F1=F1, F05=F05, lo=lo, hi=hi)

# ── sampling knowledge resource ──────────────────────────────────────────────
def sample_dict(d, frac, seed):
    if frac >= 1.0: return dict(d)
    if frac <= 0.0: return {}
    keys = sorted(d)
    random.Random(seed).shuffle(keys)
    return {w: d[w] for w in keys[:int(round(len(keys)*frac))]}

# ── run ──────────────────────────────────────────────────────────────────────
ABL_CONFIGS = [  # (label, use_lookup, use_leet, use_censor)
    ("lookup only",          True,  False, False),
    ("leet only",            False, True,  False),
    ("censor only",          False, False, True),
    ("lookup+leet",          True,  True,  False),
    ("lookup+censor",        True,  False, True),
    ("leet+censor",          False, True,  True),
    ("full (all three)",     True,  True,  True),
]

def run(args):
    data = json.load(open(args.test, encoding="utf-8"))
    vocab = load_validity_dict(args.dict)
    base = len(vocab)
    added = augment_vocab_from_splits(vocab, [TRAIN, VAL])
    lookup = build_lookup(args.algo)
    print(f"vocab base={base} +aug={added} → {len(vocab)} | lookup={len(lookup)} entri "
          f"| test files={len(data)}")

    rows = []
    # (1) ablation per-komponen
    print("\n=== (1) ABLATION KOMPONEN (scope ALG, test) ===")
    m = point_and_ci(eval_sentences(lambda t: old_leet(t), data), args.boot)
    print(f"  {'old blind leet (baseline)':26s} P={m['P']*100:5.1f} R={m['R']*100:5.1f} "
          f"F0.5={m['F05']*100:5.1f} [{m['lo']*100:.1f},{m['hi']*100:.1f}] "
          f"(tp{m['tp']} fp{m['fp']} fn{m['fn']})")
    rows.append(dict(exp="ablation", config="old_blind_leet", frac=1.0, seed=-1, **m))
    for label, ul, ule, uc in ABL_CONFIGS:
        nz = ComponentNormalizer(vocab, lookup, use_lookup=ul, use_leet=ule, use_censor=uc)
        m = point_and_ci(eval_sentences(lambda t: nz.norm_token(t)[0], data), args.boot)
        print(f"  {label:26s} P={m['P']*100:5.1f} R={m['R']*100:5.1f} "
              f"F0.5={m['F05']*100:5.1f} [{m['lo']*100:.1f},{m['hi']*100:.1f}] "
              f"(tp{m['tp']} fp{m['fp']} fn{m['fn']})")
        rows.append(dict(exp="ablation", config=label.replace(" ", "_"), frac=1.0, seed=-1, **m))

    # (2) kurva knowledge
    fracs = [float(x)/100.0 for x in args.fracs.split(",")]
    for target in ("lookup", "vocab"):
        print(f"\n=== (2) KURVA: F0.5 vs fraksi {target} "
              f"({'lookup dictionary' if target=='lookup' else 'validity lexicon'}) ===")
        for frac in fracs:
            seeds = [0] if frac in (0.0, 1.0) else list(range(args.seeds))
            f05s = []
            for sd in seeds:
                lk = sample_dict(lookup, frac, sd) if target == "lookup" else lookup
                vc = sample_dict(vocab, frac, sd) if target == "vocab" else vocab
                nz = ComponentNormalizer(vc, lk)
                m = point_and_ci(eval_sentences(lambda t: nz.norm_token(t)[0], data), boot=0)
                f05s.append(m["F05"])
                rows.append(dict(exp=f"curve_{target}", config="full", frac=frac, seed=sd, **m))
            mean = sum(f05s)/len(f05s)
            sd_ = (sum((x-mean)**2 for x in f05s)/len(f05s))**0.5 if len(f05s) > 1 else 0.0
            print(f"  frac={frac*100:5.1f}%  F0.5={mean*100:5.1f} ± {sd_*100:4.1f}  (n={len(f05s)})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n→ {args.out} ({len(rows)} baris)")

# ── selftest (tanpa file server) ─────────────────────────────────────────────
def selftest():
    vocab = {w: 100 for w in ["sadar", "ponsel", "pembunuh", "asusila", "dari", "yang", "koper"]}
    lookup = {"dr": "dari", "yg": "yang"}
    mkdata = [dict(sentences=[
        dict(teks_ocr="dr sad4r as*sila koper",
             errors=[dict(token_salah="dr",      token_benar="dari",    category="ALG"),
                     dict(token_salah="sad4r",   token_benar="sadar",   category="ALG"),
                     dict(token_salah="as*sila", token_benar="asusila", category="ALG")]),
        dict(teks_ocr="p3mbvnvh yg ponsel",
             errors=[dict(token_salah="p3mbvnvh", token_benar="pembunuh", category="ALG"),
                     dict(token_salah="yg",       token_benar="yang",     category="ALG")]),
    ])]
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}")
        ok = ok and cond

    full = ComponentNormalizer(vocab, lookup)
    m = point_and_ci(eval_sentences(lambda t: full.norm_token(t)[0], mkdata), boot=50)
    check("full: semua 5 error terkoreksi (tp=5 fp=0 fn=0)", (m["tp"], m["fp"], m["fn"]) == (5, 0, 0))
    check("full: F0.5=1.0 dan CI valid", abs(m["F05"]-1.0) < 1e-9 and m["lo"] <= m["F05"] <= m["hi"])

    lo_ = ComponentNormalizer(vocab, lookup, use_leet=False, use_censor=False)
    m = point_and_ci(eval_sentences(lambda t: lo_.norm_token(t)[0], mkdata), boot=0)
    check("lookup-only: hanya dr,yg (tp=2 fn=3)", (m["tp"], m["fn"]) == (2, 3))

    le_ = ComponentNormalizer(vocab, lookup, use_lookup=False, use_censor=False)
    check("leet-only: sad4r ✓", le_.norm_token("sad4r")[0] == "sadar")
    check("leet-only: as*sila ✗ (censor off)", le_.norm_token("as*sila")[0] == "as*sila")
    check("leet-only: dr ✗ (lookup off)", le_.norm_token("dr")[0] == "dr")

    ce_ = ComponentNormalizer(vocab, lookup, use_lookup=False, use_leet=False)
    check("censor-only: as*sila ✓", ce_.norm_token("as*sila")[0] == "asusila")
    check("censor-only: p3mbvnvh ✗ (leet off)", ce_.norm_token("p3mbvnvh")[0] == "p3mbvnvh")

    check("sample frac=0 → kosong", sample_dict(lookup, 0.0, 0) == {})
    check("sample frac=1 → utuh", sample_dict(lookup, 1.0, 0) == lookup)
    s1, s2 = sample_dict(vocab, 0.5, 1), sample_dict(vocab, 0.5, 2)
    check("sample 50% ≈ setengah & seed-dependent",
          abs(len(s1) - len(vocab)/2) <= 1 and (s1 != s2 or len(vocab) <= 2))
    empty = ComponentNormalizer({}, lookup)
    check("vocab=0%: leet mati (sad4r tak berubah)", empty.norm_token("sad4r")[0] == "sad4r")
    print("\n=== SELF-TEST", "LULUS ===" if ok else "ADA FAIL ===")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--fracs", default="0,25,50,75,100")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--algo", default=ALGO)
    ap.add_argument("--dict", default=DICTF)
    ap.add_argument("--test", default=TEST)
    ap.add_argument("--out", default="results/knowledge_curve.csv")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.run:
        run(args); return
    ap.error("pilih --selftest | --run")

if __name__ == "__main__":
    main()
