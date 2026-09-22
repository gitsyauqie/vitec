#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX T6 — Rebuild data DA dengan WordPiece + tag BENAR (anti-UNK).

Bug yang diperbaiki: data DA lama = teks PLAIN dibinarisasi ke dict WordPiece
→ 92% token jadi <unk>. Skrip ini meng-WordPiece-kan teks (tokenizer.json yang
sama dgn pretrain) sehingga UNK ~0%, dan memetakan tag kategori ke 3 tag yang
DIKENAL model (sesuai prepare_syn.py), bukan tag pendek <alg>/<eja>/... yang salah.

Output (ke OUT_DIR):
  train.wp.src / train.wp.tgt   (526 real + N_SYN syn, 1:3, WordPiece, tag benar)
  valid.wp.src / valid.wp.tgt   (real val, WordPiece)
  test_c2.wp.src / .wp.tgt + test_c2.meta.jsonl   (src = transkripsi teks_ocr → C2 oracle)
  test_c1.wp.src / .wp.tgt + test_c1.meta.jsonl   (src = paddleocr_out → C1 real OCR)

Self-check: assert UNK-rate < 1% di semua file src (kalau gagal → berhenti, jgn lanjut train).

PATH: semua path berasal dari env VITEC_ROOT (default = path server aslinya), jadi
skrip tetap jalan apa adanya di server dan bisa dipindah dengan
`export VITEC_ROOT=/path/anda`. Lihat README §Configuration.
"""
import csv, json, os, random, sys
from collections import Counter

random.seed(42)
ROOT      = os.environ.get("VITEC_ROOT", "/ssd-data1/sq2023/VITEC")
CASTLE    = os.environ.get("CASTLE_ROOT", "/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")
OUT_DIR   = f"{ROOT}/data/real_da_wp"
SYN_DIR   = f"{ROOT}/data/syn"                       # plain syn (train.src/.tgt) — sudah tag benar
TRAIN_JSON= f"{ROOT}/data/splits/train.json"
VAL_JSON  = f"{ROOT}/data/splits/val.json"
OCR_CSV   = f"{ROOT}/results/e2e_ocr_outputs.csv"    # crop-level test (201): teks_ocr, paddleocr_out, kalimat_koreksi, category
TOKENIZER = os.environ.get("VITEC_TOKENIZER", f"{CASTLE}/raw-wordpiece/wordpiece_tokenizer/tokenizer.json")
SRC_DICT  = f"{ROOT}/data/processed/itiec_syn_wp20k_bin/dict.source.txt"
TGT_DICT  = f"{ROOT}/data/processed/itiec_syn_wp20k_bin/dict.target.txt"
SYN_RATIO = 3      # real:syn = 1:3
SPECIAL   = ("[CLS]", "[SEP]")

# Kode kategori (real) / nama (syn) → 3 tag yang dikenal model (sama dgn prepare_syn.py)
TAG_MAP = {
    "ALG": "<semantic>",     "algospeak":    "<semantic>",
    "SEM": "<semantic>",     "semantic":     "<semantic>",
    "SIN": "<syntactic>",    "syntactic":    "<syntactic>",
    "EJA": "<morphological>","spelling":     "<morphological>",
    "MOR": "<morphological>","morphological":"<morphological>",
    "TIP": "<morphological>","typographic":  "<morphological>",
    "MIX": "<morphological>",
}
DEFAULT_TAG = "<morphological>"

# ── tokenizer ────────────────────────────────────────────────────────────────
from tokenizers import Tokenizer
_tok = Tokenizer.from_file(TOKENIZER)

def wp(text: str) -> str:
    """WordPiece-tokenize satu baris, buang [CLS]/[SEP], join spasi."""
    toks = [t for t in _tok.encode(text).tokens if t not in SPECIAL]
    return " ".join(toks)

def sent_tag(sentence: dict, file_primary: str) -> str:
    """Tag per kalimat dari errors[].category mayoritas; fallback primary_category."""
    cats = [e.get("category", "").upper() for e in sentence.get("errors", []) if e.get("category")]
    code = Counter(cats).most_common(1)[0][0] if cats else (file_primary or "").upper()
    return TAG_MAP.get(code, DEFAULT_TAG)

# ── load dict utk cek UNK ──────────────────────────────────────────────────────
def load_dict(path):
    return set(l.split()[0] for l in open(path, encoding="utf-8"))

SRC_VOCAB = load_dict(SRC_DICT)

def unk_rate(wp_lines):
    """Rate token di luar dict + Counter token OOV (utk transparansi)."""
    tot = unk = 0
    oov = Counter()
    for l in wp_lines:
        for w in l.split():
            tot += 1
            if w not in SRC_VOCAB:
                unk += 1
                oov[w] += 1
    return (unk / tot if tot else 0.0), tot, oov

# ── builders ───────────────────────────────────────────────────────────────────
def build_real(json_path):
    """Kembalikan list (src_wp, tgt_wp) dari split JSON. src = teks_ocr (transkripsi error)."""
    data = json.load(open(json_path, encoding="utf-8"))
    pairs = []
    for f in data:
        prim = f.get("primary_category", "")
        for s in f.get("sentences", []):
            src = (s.get("teks_ocr") or "").strip()
            tgt = (s.get("kalimat_koreksi") or "").strip()
            if not src or not tgt:
                continue
            tag = sent_tag(s, prim)
            pairs.append((wp(f"{tag} {src}"), wp(tgt)))
    return pairs

def build_syn(n):
    """Ambil n pair syn dari data/syn (plain, tag sudah benar) → WordPiece."""
    src_l = [l.rstrip("\n") for l in open(f"{SYN_DIR}/train.src", encoding="utf-8")]
    tgt_l = [l.rstrip("\n") for l in open(f"{SYN_DIR}/train.tgt", encoding="utf-8")]
    idx = list(range(len(src_l)))
    random.shuffle(idx)
    idx = idx[:n]
    return [(wp(src_l[i]), wp(tgt_l[i])) for i in idx]

def build_eval_from_csv(src_col):
    """Eval set dari e2e_ocr_outputs.csv. Return (rows_wp_src, rows_wp_tgt, meta)."""
    rows = list(csv.DictReader(open(OCR_CSV, encoding="utf-8")))
    src_wp, tgt_wp, meta = [], [], []
    for i, r in enumerate(rows):
        cat = (r.get("category") or "").upper()
        tag = TAG_MAP.get(cat, DEFAULT_TAG)
        src = (r.get(src_col) or "").strip()
        ref = (r.get("kalimat_koreksi") or "").strip()
        src_wp.append(wp(f"{tag} {src}"))
        tgt_wp.append(wp(ref))                      # placeholder utk fairseq-generate
        meta.append({"id": i, "category": cat, "src_orig": src, "ref_orig": ref})
    return src_wp, tgt_wp, meta

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

UNK_THRESHOLD = 0.05   # <5% = OOV domain wajar (real social-media). >5% = curiga bug.
def check_unk(name, src_lines):
    rate, tot, oov = unk_rate(src_lines)
    flag = "OK" if rate < UNK_THRESHOLD else "‼️ TINGGI (curiga bug)"
    print(f"  [{name}] OOV={rate*100:.3f}%  ({tot} tok)  {flag}")
    if oov:
        print(f"         top OOV: {oov.most_common(8)}")
    return rate

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("== Build REAL train ==")
    real_tr = build_real(TRAIN_JSON)
    print(f"  real train pairs: {len(real_tr)}")
    n_syn = len(real_tr) * SYN_RATIO
    syn_tr = build_syn(n_syn)
    print(f"  syn pairs (1:{SYN_RATIO}): {len(syn_tr)}")
    train = real_tr + syn_tr
    random.shuffle(train)
    tr_src = [s for s, _ in train]; tr_tgt = [t for _, t in train]

    print("== Build REAL valid ==")
    val = build_real(VAL_JSON)
    va_src = [s for s, _ in val]; va_tgt = [t for _, t in val]
    print(f"  valid pairs: {len(val)}")

    print("== Build eval sets dari OCR CSV ==")
    c2_src, c2_tgt, c2_meta = build_eval_from_csv("teks_ocr")       # C2 oracle transcription
    c1_src, c1_tgt, c1_meta = build_eval_from_csv("paddleocr_out")  # C1 real OCR
    print(f"  C2 (transkripsi) lines: {len(c2_src)} | C1 (OCR) lines: {len(c1_src)}")

    # tulis
    write_lines(f"{OUT_DIR}/train.wp.src", tr_src); write_lines(f"{OUT_DIR}/train.wp.tgt", tr_tgt)
    write_lines(f"{OUT_DIR}/valid.wp.src", va_src); write_lines(f"{OUT_DIR}/valid.wp.tgt", va_tgt)
    write_lines(f"{OUT_DIR}/test_c2.wp.src", c2_src); write_lines(f"{OUT_DIR}/test_c2.wp.tgt", c2_tgt)
    write_lines(f"{OUT_DIR}/test_c1.wp.src", c1_src); write_lines(f"{OUT_DIR}/test_c1.wp.tgt", c1_tgt)
    write_meta(f"{OUT_DIR}/test_c2.meta.jsonl", c2_meta)
    write_meta(f"{OUT_DIR}/test_c1.meta.jsonl", c1_meta)

    print("== Self-check UNK (harus <1%) ==")
    rates = [
        check_unk("train", tr_src), check_unk("valid", va_src),
        check_unk("test_c2", c2_src), check_unk("test_c1", c1_src),
    ]
    print(f"\n→ Output di {OUT_DIR}")
    if max(rates) >= UNK_THRESHOLD:
        print(f"‼️  OOV >{UNK_THRESHOLD*100:.0f}% — curiga bug sistematis (cek top OOV: tag ter-split?). STOP.")
        sys.exit(1)
    print(f"✅ OOV <{UNK_THRESHOLD*100:.0f}% (domain noise wajar; turun drastis dari 92%). Lanjut: preprocess_real_da.sh")

if __name__ == "__main__":
    main()
