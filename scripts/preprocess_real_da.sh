#!/bin/bash
# FIX T6 — binarisasi data DA WordPiece pakai dict yg SAMA dgn castle_ft_iged_newarch.
set -e
ROOT=${VITEC_ROOT:-/ssd-data1/sq2023/VITEC}
WP=$ROOT/data/real_da_wp
SRCDICT=$ROOT/data/processed/itiec_syn_wp20k_bin/dict.source.txt
TGTDICT=$ROOT/data/processed/itiec_syn_wp20k_bin/dict.target.txt

# 1) train + valid → satu bin dir
fairseq-preprocess \
  --source-lang src --target-lang tgt \
  --trainpref "$WP/train.wp" --validpref "$WP/valid.wp" \
  --srcdict "$SRCDICT" --tgtdict "$TGTDICT" \
  --destdir "$ROOT/data/processed/itiec_real_da_wp_bin" \
  --workers 8

# 2) test C2 (transkripsi oracle) → gen-subset 'test'
fairseq-preprocess \
  --source-lang src --target-lang tgt \
  --testpref "$WP/test_c2.wp" \
  --srcdict "$SRCDICT" --tgtdict "$TGTDICT" \
  --destdir "$ROOT/data/processed/da_test_c2_bin" \
  --workers 4

# 3) test C1 (real OCR) → gen-subset 'test'
fairseq-preprocess \
  --source-lang src --target-lang tgt \
  --testpref "$WP/test_c1.wp" \
  --srcdict "$SRCDICT" --tgtdict "$TGTDICT" \
  --destdir "$ROOT/data/processed/da_test_c1_bin" \
  --workers 4

echo "✅ Preprocess selesai:"
echo "   train/valid → itiec_real_da_wp_bin   (cek log: '<unk>' harus ~0)"
echo "   C2 → da_test_c2_bin | C1 → da_test_c1_bin"
