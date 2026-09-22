#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE B — L2 Orthography Normalizer (presisi tinggi, deterministik, tanpa halusinasi).

Scope (sesuai gold errors[]): kategori ALG = leet/number-substitution + kata tersensor (*),
plus singkatan umum (abbreviations). BUKAN kapitalisasi (TIP) atau tata bahasa (MOR/SIN/SEM)
— itu urusan GEC. Prinsip: hanya ubah token bila yakin (hasil = kata kamus sah / singkatan
dikenal); selain itu biarkan apa adanya.

Tiga komponen:
  1) LOOKUP   : invert algospeak_dict (abbreviations/combined/slang/error_patterns) → wrong→correct.
  2) LEET     : decode 4→a,3→e,0→o,5→s,9→g,... via candidate-generation + validasi kamus.
  3) CENSOR   : isi '*' (wildcard) + kombinasi leet → pilih kandidat yg jadi kata kamus.

Pakai:
  python L2_normalizer.py --selftest                 # verifikasi logika (mini-dict inline)
  python L2_normalizer.py --eval                     # eval standalone vs gold errors[] + ablation
  python L2_normalizer.py --normalize-csv IN.csv --col paddleocr_out --out OUT.csv  # utk pipeline
"""
import argparse, csv, json, os, re, string, sys, itertools
from collections import Counter, defaultdict

ALGO  = os.environ.get("PAPER5_ROOT","/ssd-data1/sq2023/Paper5")+"/ITIEC_Synthetic/algospeak_dict.json"
DICTF = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/processed/syn_data_bin/dict.src.txt"
TEST  = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/splits/test.json"
TRAIN = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/splits/train.json"
VAL   = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/splits/val.json"
MINFREQ = 2          # ambang frekuensi kata dianggap "sah" di kamus
PUNCT = string.punctuation

# Inversi number_substitutions → decode. Ambigu = >1 opsi.
LEET = {
    "4": ["a"], "@": ["a"], "3": ["e"], "0": ["o"], "5": ["s"], "$": ["s"],
    "7": ["t"], "8": ["b"], "9": ["g"], "2": ["z"], "1": ["i", "l"], "!": ["i"],
}
AMBIG_LETTER = {"v": ["v", "u"]}   # v sering = u (pembvnvh→pembunuh) tapi bisa tetap v
LEET_CHARS = set(LEET) | {"*"} | set(AMBIG_LETTER)

# ── dictionary & lookup ──────────────────────────────────────────────────────
def load_validity_dict(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) < 2: continue
        w, f = p[0], p[-1]
        if not w.isalpha() or len(w) < 2: continue
        try: fr = int(f)
        except: continue
        if fr >= MINFREQ:
            d[w.lower()] = fr
    return d

def augment_vocab_from_splits(vocab, paths):
    """Tambah kata sah dari token_benar train/val (leakage-free utk test).
       Pecah multi-kata jadi token; hanya alfabetik, len>=2."""
    added = 0
    for p in paths:
        if not os.path.exists(p): continue
        for f in json.load(open(p, encoding="utf-8")):
            for s in f.get("sentences", []):
                for e in s.get("errors", []):
                    bn = (e.get("token_benar") or "")
                    for w in bn.split():
                        w = w.strip(PUNCT).lower()
                        if w.isalpha() and len(w) >= 2 and w not in vocab:
                            vocab[w] = MINFREQ; added += 1
                # kalimat_koreksi juga sumber kata sah
                for w in (s.get("kalimat_koreksi") or "").split():
                    w = w.strip(PUNCT).lower()
                    if w.isalpha() and len(w) >= 2 and w not in vocab:
                        vocab[w] = MINFREQ; added += 1
    return added

def build_lookup(algo_path):
    """wrong→correct, subset AMAN (unik, bukan kata sah, len≥2)."""
    if not os.path.exists(algo_path): return {}
    a = json.load(open(algo_path, encoding="utf-8"))
    raw = defaultdict(set)   # wrong → {correct,...}
    def add(correct, variants):
        for v in (variants if isinstance(variants, list) else [variants]):
            v = str(v).strip().lower()
            if len(v) >= 2 and " " not in v:
                raw[v].add(correct.strip().lower())
    for sec in ("abbreviations", "phonetic_substitutions", "instagram_tiktok_slang"):
        for correct, vars_ in a.get(sec, {}).items():
            add(correct, vars_)
    for rule in a.get("combined_rules", []):
        add(rule["from"], rule.get("to", []))
    for pat in a.get("error_patterns", {}).values():
        for ex in pat.get("examples", []):
            frm, to = ex.get("from"), ex.get("to")
            if isinstance(to, list):
                for t in to: add(frm, t)
            elif to: add(frm, to)
    # hanya simpan yg unik (1 target) → presisi
    return {w: list(cs)[0] for w, cs in raw.items() if len(cs) == 1}

# ── core normalizer ──────────────────────────────────────────────────────────
class Normalizer:
    def __init__(self, vocab, lookup, max_combos=4000):
        self.vocab = vocab            # word→freq
        self.lookup = lookup
        self.max_combos = max_combos

    def _candidates(self, core):
        opts = []
        for ch in core:
            if ch in LEET:           opts.append(LEET[ch])
            elif ch == "*":          opts.append(list(string.ascii_lowercase))
            elif ch in AMBIG_LETTER: opts.append(AMBIG_LETTER[ch])
            else:                    opts.append([ch])
        total = 1
        for o in opts: total *= len(o)
        if total > self.max_combos: return []
        return ["".join(p) for p in itertools.product(*opts)]

    def norm_token(self, tok):
        """return (new_token, fired, method)."""
        lead = tok[:len(tok) - len(tok.lstrip(PUNCT))]
        trail = tok[len(tok.rstrip(PUNCT)):]
        core = tok[len(lead): len(tok) - len(trail)]
        low = core.lower()
        if not core:
            return tok, False, None
        # angka murni / ada digit tapi tak ada huruf → biarkan (mis. 100, 3.1, 1,5)
        if not any(c.isalpha() for c in core):
            return tok, False, None
        # 1) lookup singkatan/slang (hanya kalau bukan kata sah)
        if low in self.lookup and low not in self.vocab:
            return lead + self.lookup[low] + trail, True, "lookup"
        has_special = any(c in LEET_CHARS for c in low)
        # 2) sudah kata sah & tak ada karakter leet/sensor → jangan sentuh
        if low in self.vocab and not any(c in LEET or c == "*" for c in low):
            return tok, False, None
        # 3) leet / censor decode
        if has_special and any(c in LEET or c == "*" or c in AMBIG_LETTER for c in low):
            cands = self._candidates(low)
            valid = [(c, self.vocab[c]) for c in cands if c in self.vocab]
            if valid:
                best = max(valid, key=lambda x: x[1])[0]
                if best != low:
                    return lead + best + trail, True, "leet/censor"
        return tok, False, None

    def norm_text(self, text):
        out = []
        for t in text.split():
            nt, _, _ = self.norm_token(t)
            out.append(nt)
        return " ".join(out)

# baseline lama: blind leet 4→a,3→e,0→o ke semua token
_OLD = str.maketrans({"4": "a", "3": "e", "0": "o"})
def old_leet(text):
    return text.translate(_OLD)

# ── eval standalone vs gold errors[] ─────────────────────────────────────────
def normcmp(a, b):
    return a.strip(PUNCT).lower() == b.strip(PUNCT).lower()

def evaluate(norm: "Normalizer", test_path):
    data = json.load(open(test_path, encoding="utf-8"))
    # metrik: scope ALG. clean token = tak muncul di error apa pun.
    def run(fn, scope=("ALG",)):
        tp = fp = fn_ = 0
        for f in data:
            for s in f.get("sentences", []):
                src = s.get("teks_ocr", "")
                gold = {}      # core(salah) → benar  (utk scope)
                allerr = set() # core(salah) semua kategori
                for e in s.get("errors", []):
                    sl = (e.get("token_salah") or "").strip()
                    bn = (e.get("token_benar") or "").strip()
                    if not sl: continue
                    allerr.add(sl.strip(PUNCT).lower())
                    if e.get("category") in scope and " " not in sl:
                        gold[sl.strip(PUNCT).lower()] = bn
                for tok in src.split():
                    core = tok.strip(PUNCT).lower()
                    pred = fn(tok)
                    changed = (pred.strip(PUNCT).lower() != core)
                    if core in gold:
                        if normcmp(pred, gold[core]): tp += 1
                        else: fn_ += 1
                    elif core not in allerr:           # token bersih
                        if changed: fp += 1
        p = tp/(tp+fp) if (tp+fp) else 0.0
        r = tp/(tp+fn_) if (tp+fn_) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        f05 = 1.25*p*r/(0.25*p+r) if (0.25*p+r) else 0.0
        return dict(tp=tp, fp=fp, fn=fn_, P=p, R=r, F1=f1, F05=f05)

    print("=== EVAL L2 (scope ALG; clean-token over-correction = FP) ===")
    for name, fn in [("OLD leet (4a3e0o blind)", lambda t: old_leet(t)),
                     ("L2 normalizer (full)",     lambda t: norm.norm_token(t)[0])]:
        m = run(fn)
        print(f"  {name:28s} P={m['P']*100:5.1f}  R={m['R']*100:5.1f}  "
              f"F1={m['F1']*100:5.1f}  F0.5={m['F05']*100:5.1f}  (tp{m['tp']} fp{m['fp']} fn{m['fn']})")

    # contoh kualitatif
    print("\n=== Contoh normalisasi (token ALG/leet/sensor) ===")
    seen = 0
    for f in data:
        for s in f.get("sentences", []):
            for e in s.get("errors", []):
                if e.get("category") != "ALG": continue
                sl = (e.get("token_salah") or "").strip()
                if " " in sl or not sl: continue
                pred = norm.norm_token(sl)[0]
                mark = "✓" if normcmp(pred, e.get("token_benar","")) else "✗"
                print(f"  {mark} {sl:14s} → {pred:14s} (gold: {e.get('token_benar')})")
                seen += 1
                if seen >= 20: return
    return

# ── self-test (mini dict inline, tanpa file server) ──────────────────────────
def selftest():
    vocab = {w: 100 for w in ["sadar","hilang","ponsel","pembunuh","ditemukan",
                              "asusila","selingkuh","dari","celana","mabuk","video","vicky"]}
    lookup = {"dr": "dari", "yg": "yang"}
    nz = Normalizer(vocab, lookup)
    cases = [
        ("sad4r","sadar"), ("hil4ng","hilang"), ("P0nsel","ponsel"),
        ("p3mbvnvh","pembunuh"), ("ditemvkan","ditemukan"),
        ("as*sila","asusila"), ("dr","dari"),
        ("video","video"),      # kata sah → jangan diubah
        ("vicky","vicky"),      # nama, v tetap (tak ada kata 'uicky')
        ("100","100"), ("3.1","3.1"),  # angka → jangan disentuh
    ]
    ok = True
    for inp, exp in cases:
        got = nz.norm_token(inp)[0]
        flag = "OK" if got.lower()==exp.lower() else "FAIL"
        if flag=="FAIL": ok=False
        print(f"  [{flag}] {inp:12s} → {got:12s} (exp {exp})")
    print("\n=== SELF-TEST", "LULUS ===" if ok else "ADA FAIL ===")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--normalize-csv")
    ap.add_argument("--col", default="paddleocr_out")
    ap.add_argument("--out")
    ap.add_argument("--algo", default=ALGO)
    ap.add_argument("--dict", default=DICTF)
    ap.add_argument("--test", default=TEST)
    ap.add_argument("--no-aug", action="store_true", help="jangan augmentasi kamus dari train/val")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return

    vocab = load_validity_dict(args.dict)
    base = len(vocab)
    if not args.no_aug:
        added = augment_vocab_from_splits(vocab, [TRAIN, VAL])
        print(f"vocab base={base} + aug(train/val)={added} → {len(vocab)} kata")
    lookup = build_lookup(args.algo)
    print(f"vocab={len(vocab)} kata (freq≥{MINFREQ}) | lookup={len(lookup)} entri singkatan")
    nz = Normalizer(vocab, lookup)

    if args.eval:
        evaluate(nz, args.test); return

    if args.normalize_csv:
        rows = list(csv.DictReader(open(args.normalize_csv, encoding="utf-8")))
        for r in rows:
            r["normalized"] = nz.norm_text(r.get(args.col, "") or "")
        outp = args.out or args.normalize_csv.replace(".csv", "_norm.csv")
        with open(outp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"→ {outp} (kolom 'normalized' ditambahkan)")
        return
    ap.error("pilih --selftest | --eval | --normalize-csv")

if __name__ == "__main__":
    main()
