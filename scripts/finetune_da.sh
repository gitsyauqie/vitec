#!/bin/bash
# FIX T6 — re-finetune DA dari castle_ft_iged_newarch pada data WordPiece yg BENAR.
# Config identik log castle_da_nokg (semantic_weight=0), HANYA data-bin yg berubah (anti-UNK).
#
# PATH: override dengan env VITEC_ROOT / CASTLE_ROOT. Default = path server asli.
set -e
ROOT=${VITEC_ROOT:-/ssd-data1/sq2023/VITEC}
CASTLE=${CASTLE_ROOT:-/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6}
EXT=$CASTLE/fairseq_extensions
KG=$CASTLE/semantic_kg.json
DATA=$ROOT/data/processed/itiec_real_da_wp_bin
RESTORE=$ROOT/models/checkpoints/castle_ft_iged_newarch/checkpoint_best.pt
SAVE=$ROOT/models/checkpoints/castle_da_nokg_wp
GPU=${1:-4}     # pakai: bash finetune_da.sh <gpu_id>

mkdir -p "$SAVE" "$ROOT/logs"
CUDA_VISIBLE_DEVICES=$GPU fairseq-train "$DATA" \
  --user-dir "$EXT" \
  --arch castle_transformer --task translation \
  --save-dir "$SAVE" \
  --restore-file "$RESTORE" \
  --reset-optimizer --reset-lr-scheduler --reset-meters --reset-dataloader \
  --semantic-weight 0.0 --kg-path "$KG" \
  --share-decoder-input-output-embed \
  --encoder-layers 4 --decoder-layers 4 \
  --encoder-embed-dim 256 --decoder-embed-dim 256 \
  --encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 8 --decoder-attention-heads 8 \
  --dropout 0.3 --attention-dropout 0.1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
  --lr 3e-05 --lr-scheduler inverse_sqrt --warmup-updates 200 --warmup-init-lr 1e-07 \
  --max-tokens 4096 --update-freq 2 \
  --max-epoch 40 --patience 10 \
  --best-checkpoint-metric loss --keep-best-checkpoints 2 \
  --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test \
  --log-format json --log-interval 50 \
  2>&1 | tee "$ROOT/logs/castle_da_nokg_wp.log"

echo "✅ Train selesai → $SAVE/checkpoint_best.pt"
echo "   CEK: nll_loss akhir harus JAUH < 3.68 (target mendekati ~0.3 spt syn) bila bug UNK benar penyebabnya."
