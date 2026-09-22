#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siapkan set VALIDASI untuk memilih ambang gating tau (menutup kritik
"operating point selected on the evaluation split").

Tidak perlu menjalankan ulang OCR: results/train_ocr_pairs.csv sudah berisi
pasangan (gt, ocr) untuk crop TRAIN **dan** VAL (lihat run_ocr_train.py).
Skrip ini menyaring baris milik split val, menormalisasi dengan L2, lalu menulis
val_c3.wp.{src,tgt} + meta dengan format identik prep_c3.py.

Alur lengkap (server, env fairseq):

  cd /ssd-data1/sq2023/VITEC/scripts

  # 1) bangun set validasi C3 (OCR -> L2)
  python prep_val_gate.py

  # 2) binarisasi dengan dict yang sama
  fairseq-preprocess --source-lang src --target-lang tgt \
    --testpref /ssd-data1/sq2023/VITEC/data/real_da_wp/val_c3.wp \
    --srcdict /ssd-data1/sq2023/VITEC/data/processed/itiec_syn_wp20k_bin/dict.source.txt \
    --tgtdict /ssd-data1/sq2023/VITEC/data/processed/itiec_syn_wp20k_bin/dict.target.txt \
    --destdir /ssd-data1/sq2023/VITEC/data/processed/da_val_c3_bin --workers 4

  # 3) generate dengan checkpoint yang sama dengan hasil utama
  python gen_eval_da.py \
    --bin  /ssd-data1/sq2023/VITEC/data/processed/da_val_c3_bin \
    --meta /ssd-data1/sq2023/VITEC/data/real_da_wp/val_c3.meta.jsonl \
    --ckpt /ssd-data1/sq2023/VITEC/models/checkpoints/castle_noise_matched/checkpoint_best.pt \
    --out  /ssd-data1/sq2023/VITEC/results/val_C3_nm2.csv --gpu 4

  # 4) pilih tau DI VALIDASI (ambil tau dari baris BEST, catat nilainya)
  python selective_apply.py --pred /ssd-data1/sq2023/VITEC/results/val_C3_nm2.csv \
    --cer-norm lower_nopunct

  # 5) terapkan tau itu SEKALI ke test -- tanpa sweep lagi
  python apply_fixed_tau.py --pred /ssd-data1/sq2023/VITEC/results/e2e_C3_nm2.csv \
    --tau <NILAI_DARI_LANGKAH_4> --cer-norm lower_nopunct
"""
import csv, json, os, sys

ROOT = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+""
OUT = f"{ROOT}/data/real_da_wp"
PAIRS = f"{ROOT}/results/train_ocr_pairs.csv"
VAL = f"{ROOT}/data/splits/val.json"
TOKENIZER = (os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/raw-wordpiece/"
             "wordpiece_tokenizer/tokenizer.json")

TAG_MAP = {"ALG": "<semantic>", "SEM": "<semantic>", "SIN": "<syntactic>",
           "EJA": "<morphological>", "MOR": "<morphological>",
           "TIP": "<morphological>", "MIX": "<morphological>"}
DEFAULT = "<morphological>"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from L2_normalizer import (Normalizer, load_validity_dict,
                           augment_vocab_from_splits, build_lookup,
                           ALGO, DICTF, TRAIN, VAL as VAL_SPLIT)


def main():
    # ── referensi + kategori untuk crop validasi ──────────────────────────────
    ref, cat = {}, {}
    for e in json.load(open(VAL, encoding="utf-8")):
        base = os.path.splitext(e["filename"])[0]
        for s in e.get("sentences", []):
            key = f"{base}_{s['sentence_id']}"
            k = (s.get("kalimat_koreksi") or "").strip()
            if k:
                ref[key] = k
                cat[key] = (s.get("category") or e.get("primary_category") or "").upper()
    print(f"kalimat validasi ber-referensi: {len(ref)}")

    # ── OCR validasi diambil dari pasangan train+val yang sudah ada ───────────
    rows = []
    for r in csv.DictReader(open(PAIRS, encoding="utf-8")):
        key = os.path.splitext(os.path.basename(r["crop"]))[0]
        if key in ref and (r.get("ocr") or "").strip():
            rows.append((key, r["ocr"].strip()))
    print(f"crop validasi dengan output OCR: {len(rows)}")
    if not rows:
        raise SystemExit("tidak ada baris validasi di train_ocr_pairs.csv -- "
                         "cek bahwa run_ocr_train.py memang menyertakan val.json")

    # ── normalisasi L2 (konfigurasi identik dengan pipeline utama) ────────────
    vocab = load_validity_dict(DICTF)
    augment_vocab_from_splits(vocab, [TRAIN, VAL_SPLIT])
    nz = Normalizer(vocab, build_lookup(ALGO))

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)
    def wp(s):
        return " ".join(x for x in tok.encode(s).tokens if x not in ("[CLS]", "[SEP]"))

    src, tgt, meta = [], [], []
    for i, (key, ocr) in enumerate(rows):
        norm = nz.norm_text(ocr)
        c = cat.get(key, "")
        src.append(wp(f"{TAG_MAP.get(c, DEFAULT)} {norm}"))
        tgt.append(wp(ref[key]))
        meta.append({"id": i, "category": c, "src_orig": norm, "ref_orig": ref[key]})

    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/val_c3.wp.src", "w", encoding="utf-8").write("\n".join(src) + "\n")
    open(f"{OUT}/val_c3.wp.tgt", "w", encoding="utf-8").write("\n".join(tgt) + "\n")
    with open(f"{OUT}/val_c3.meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"→ {OUT}/val_c3.wp.{{src,tgt}} + meta ({len(src)} baris)")


if __name__ == "__main__":
    main()
