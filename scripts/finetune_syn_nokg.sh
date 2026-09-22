#!/bin/bash
# Item-2 KG ablation: fine-tune varian NO-KG (semantic_weight=0) di ITIEC-Syn,
# recipe identik castle_ft_iged_newarch (yg semantic_weight=0.9). Bandingkan F0.5 syn-test.
#
# PATH: override dengan env VITEC_ROOT / CASTLE_ROOT. Default = path server asli.
set -e
ROOT=${VITEC_ROOT:-/ssd-data1/sq2023/VITEC}
CASTLE=${CASTLE_ROOT:-/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6}
EXT=$CASTLE/fairseq_extensions
KG=$CASTLE/semantic_kg.json
DATA=$ROOT/data/processed/itiec_syn_wp20k_bin
RESTORE=$ROOT/models/pretrained/castle_iged_newarch/checkpoint_best.pt
SAVE=$ROOT/models/checkpoints/castle_ft_iged_nokg
GPU=${1:-4}
mkdir -p "$SAVE" "$ROOT/logs"
CUDA_VISIBLE_DEVICES=$GPU fairseq-train "$DATA" \
  --user-dir "$EXT" --arch castle_transformer --task translation \
  --save-dir "$SAVE" --restore-file "$RESTORE" \
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
  --lr 5e-04 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --max-tokens 4096 --update-freq 2 --max-epoch 30 --patience 5 \
  --best-checkpoint-metric loss --keep-best-checkpoints 1 \
  --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test \
  --log-format json --log-interval 100 \
  2>&1 | tee "$ROOT/logs/castle_ft_iged_nokg.log"
echo "✅ → $SAVE/checkpoint_best.pt"
