# Reproduction map

Every number, table, and figure in the paper, mapped to the script that produces it,
the data it consumes, and the exact command. Grouped by what a reader can run.

**Metric conventions.** End-to-end quality is CER against the corrected reference under
`--cer-norm lower_nopunct` (lower-cased, punctuation removed). Correction stages are
scored with edit-based F₀.₅ in the M² tradition. Headline numbers carry 95% bootstrap
CIs over 1,000 sentence resamples; paired tests use 5,000 resamples.

**Two aggregations — read this before comparing numbers across tables.**

| Aggregation | Definition | Produced by | Used in |
|---|---|---|---|
| corpus-level (micro) | Σ edit distance ÷ Σ reference length | `eval_protocol_v2.py` | **Tab. ladder** levels (25.93 / 24.90 / 25.43), Fig. gating |
| mean per-sentence | mean over sentences of (edit distance ÷ reference length) | `paired_test.py` | **Tab. sig** Δ values |

On the same 201 sentences these give 25.93 vs 33.06 for raw OCR: short sentences with
high relative error pull the per-sentence mean up. Paired significance testing needs
per-sentence values, so Tab. sig has to use that scale — its caption says "mean CER
change" for exactly this reason. **A Δ from Tab. sig cannot be added to a level in
Tab. ladder.** The two are consistent in direction and magnitude of difference (−1.04 pp on
both scales, coincidentally), not in absolute level.

---

## Tier 0 — runnable with nothing but Python 3.8+

No GPU, no checkpoints, no API keys, no third-party packages. These reproduce the
paper's **central claim** and its statistical support.

| Paper item | Value | Script | Command |
|---|---|---|---|
| §RQ1 (measurement) protocol correctness | 9/9 self-tests | `src/eval_protocol_v2.py` | `python src/eval_protocol_v2.py --selftest` |
| §normaliser normaliser correctness | 11/11 self-tests | `src/L2_normalizer.py` | `python src/L2_normalizer.py --selftest` |
| **Tab. ladder** row 1 — OCR only | 25.93 (CI 22.90–29.24) | `src/eval_protocol_v2.py` | `python src/eval_protocol_v2.py --pred results/e2e_ocr_outputs.csv --layer E2E --cer-norm lower_nopunct --col-hyp paddleocr_out --col-ref kalimat_koreksi` |
| **Tab. ladder** row 2 — OCR + Norm. | **24.90** (CI 21.69–28.28) | `src/eval_protocol_v2.py` | same, `--pred results/e2e_C3_nm2.csv --col-hyp source` |
| **Tab. ladder** row 4 — OCR + Norm. + Corr. | 25.43 (CI 22.41–28.77) | `src/eval_protocol_v2.py` | same, `--pred results/e2e_C3_nm2.csv --col-hyp hyp` |
| **Tab. ladder** row 3 — OCR + Corr. | 25.80 | `src/eval_protocol_v2.py` | predictions `results/e2e_C1.csv` to be added in the next release; the other three rows verify from shipped files |
| **Tab. sig** (paired significance) | −1.04 pp 45/0 *p*<0.001 · +2.23 pp 67/71 *p*=0.13 · +0.08 pp 39/38 *p*=0.80 · −0.96 pp 64/31 *p*=0.03 | `src/paired_test.py` | `python src/paired_test.py --c3 results/e2e_C3_nm2.csv --ocr results/e2e_ocr_outputs.csv --tau-pct 60` |
| **Fig. gating** (gating curve) | 25.43 / 25.58 / 25.36 / 24.97 / 24.75 / 24.64 / **24.46** / 24.70 / 24.57 / 24.68 / 24.90; oracle 23.19 | `src/selective_apply.py` | `python src/selective_apply.py --pred results/e2e_C3_nm2.csv --cer-norm lower_nopunct` |
| Threshold fixed on validation (not swept on test) | — | `src/apply_fixed_tau.py` | `python src/apply_fixed_tau.py --pred results/e2e_C3_nm2.csv --tau -0.2360 --cer-norm lower_nopunct` |

> **Why two gating scripts.** `selective_apply.py` sweeps τ over the test predictions —
> its minimum is reported in the paper as a *post hoc bound*, never as a system
> configuration. `apply_fixed_tau.py` applies a τ chosen beforehand on the validation
> split (built by `prep_val_gate.py`) and performs no sweep. The separation is
> structural, not a convention: the sweep code physically cannot produce the
> "threshold-free" number and vice versa.

---

## Tier 1 — needs the annotations (`data/ITIEC_real.json`) and split files, still no GPU

> The image-level split assignment (train/val/test) is published as a release asset
> (`ITIEC_splits.json`) alongside the test crops; `data/ITIEC_real.json` carries a
> `split` field once that file is merged.

| Paper item | Value | Script | Command |
|---|---|---|---|
| **Tab. real_dist** ITIEC-Real distribution | 653 img / 804 sent; 426/65/162 img; 526/74/204 sent | `src/prepare_real_da.py` | `python src/prepare_real_da.py` (prints split counts) |
| §dataset evaluation set | 201 of 204 (3 dropped: empty OCR or reference) | `results/e2e_ocr_outputs.csv` | `wc -l results/e2e_ocr_outputs.csv` → 202 lines incl. header |
| **Tab. l2** normaliser vs blind leet | P 78.1 / R 87.7 / F₁ 82.6 / **F₀.₅ 79.9** vs 63.1 | `src/L2_normalizer.py` | `python src/L2_normalizer.py --eval --split test` |
| **Tab. l2** component decomposition | lookup: no edit fired · leet-only 76.4 · censor-only 30.9 (P=100) | `src/knowledge_curve.py` | `python src/knowledge_curve.py --ablation` |
| **Fig. lexcurve** lexicon-coverage curve | 21.7 / 47.9 / 68.0 / 79.9 at 25/50/75/100% | `src/knowledge_curve.py` | `python src/knowledge_curve.py --curve --seeds 5` |
| §RQ2 the "120 blocked tokens" claim | all 120 lookup matches are in the validity lexicon | `src/knowledge_curve.py` | printed by `--ablation` |

---

## Tier 2 — needs GPU + fairseq + CASTLE checkpoints

Reproduces the training and inference pipeline end to end.

### Data construction

| Step | Script | Command | Output |
|---|---|---|---|
| OCR over train+val crops (leakage-free) | `src/run_ocr_train.py` | `python src/run_ocr_train.py` *(paddleocr env)* | `results/train_ocr_pairs.csv` |
| Empirical character-confusion channel | `src/build_ocr_confusion.py` | `python src/build_ocr_confusion.py --pairs results/train_ocr_pairs.csv --out results/ocr_confusion.json` | `results/ocr_confusion.json` |
| Noise-matched training pairs (**realistic** regime) | `src/gen_noisy_data.py` | `python src/gen_noisy_data.py --confusion results/ocr_confusion.json --sub-only --keep-space` | 92,768 pairs, calibrated CER 17.5% |
| Noise-matched pairs (**aggressive** ablation) | `src/gen_noisy_data.py` | same, *without* `--sub-only --keep-space` | ablation in §ablations |
| DA data → WordPiece (+ UNK self-check) | `src/prepare_real_da.py` | `python src/prepare_real_da.py` | aborts if OOV ≥ 5% |
| Held-out ITIEC-Syn test split | `src/make_syn_test.py` | `python src/make_syn_test.py --generate --iged IGED.csv` | 2,500 pairs, disjoint from train+val |
| Binarise for fairseq | `scripts/preprocess_real_da.sh` | `bash scripts/preprocess_real_da.sh` | `data/processed/*_bin` |

> **The `--sub-only --keep-space` flags are the §ablations ablation.** Aggressive noise
> (insertions, deletions, word-boundary scrambling) at the *same* overall corruption
> level teaches the model to guess: `viral→torino`, `asusila→susilo`. Restricting
> corruption to boundary-preserving substitutions eliminates those hallucinations
> entirely. This is the "noise type matters more than noise magnitude" result.

### Training

| Model | Script | Command | Paper reference |
|---|---|---|---|
| Domain-adapted corrector (post-bugfix) | `scripts/finetune_da.sh` | `bash scripts/finetune_da.sh 4` | "OCR + Corr." row, Tab. ladder |
| Noise-matched corrector | `scripts/finetune_noisy.sh` | `bash scripts/finetune_noisy.sh 4` | main pipeline corrector |
| KG ablation (semantic weight = 0) | `scripts/finetune_syn_nokg.sh` | `bash scripts/finetune_syn_nokg.sh 4` | **Tab. kg** |

### Inference and evaluation

| Paper item | Value | Command |
|---|---|---|
| **Tab. kg** KG ablation | F₀.₅ 56.4 both (w=0.9: P 55.0/R 62.9 · w=0: P 55.4/R 61.0) | `python src/gen_eval_da.py --bin data/processed/syn_test_bin --meta data/real_da_wp/syntest.meta.jsonl --ckpt <ckpt> --out results/syn_test.csv` then `eval_protocol_v2.py --layer L3` |
| §RQ3 (synthetic correction) synthetic F₀.₅ | 56.4 test / 57.4 validation | same, on the held-out test split |
| C3 pipeline predictions | → `results/e2e_C3_nm2.csv` | `python src/prep_c3.py && python src/gen_eval_da.py --bin data/processed/da_test_c3_bin --meta data/real_da_wp/test_c3.meta.jsonl --ckpt models/checkpoints/castle_noise_matched/checkpoint_best.pt --out results/e2e_C3_nm2.csv --gpu 4` |
| Validation split for τ | → `results/val_C3_nm2.csv` | `python src/prep_val_gate.py` then `gen_eval_da.py` (full recipe in the script's docstring) |

---

## Baselines (not included)

The zero-shot LLM and VLM comparisons (Tab. llm, Tab. vlm) rely on hosted, non-deterministic
APIs; their scripts and outputs are not distributed with this repository.

---

## Tier 4 — deployment benchmark

| Paper item | Value | Command |
|---|---|---|
| **Tab. deploy** normaliser | 0.02 ms median / 0.07 ms p95; 0.17 s resource load | `python src/deploy_benchmark.py --l2-csv results/e2e_ocr_outputs.csv` |
| **Tab. deploy** corrector FP32 | 86.5 MB · 343 ms · 1182 ms p95 · CER 25.5 | `python src/deploy_benchmark.py --ckpt <ckpt> --precision fp32 --threads 1 --beam 5` |
| **Tab. deploy** corrector INT8 | 66.3 MB · 215 ms · 766 ms p95 · CER 26.1 | `python src/deploy_benchmark.py --ckpt <ckpt> --precision int8` |
| **Tab. deploy** peak RAM | 823 MB | reported by the same script (`ru_maxrss`) |

Measured on a single x86 CPU thread as a conservative proxy for a mid-range mobile SoC.
Attention projections remain FP32 (framework constraint on dynamic quantisation); this
is stated in the paper rather than papered over.

---

## Numbers deliberately *not* reproducible from this repository

Listed so a reader is not left hunting for them.

| Number | Where it appeared | Status |
|---|---|---|
| F₀.₅ = 83.41% (synthetic) | earlier draft, earlier repo | **Withdrawn.** Different token-level scorer, on a synthetic split no longer part of the released corpus. Superseded by 56.4 under the released protocol. Footnoted in §RQ3 (synthetic correction) of the paper. |
| CER 33.95 / 64.08 / 39.03 | earlier draft | **Withdrawn.** Not reproducible; depended on an undocumented normalisation and OCR run. |
| "OCR CER 84–285%", E2E 13.12% | earlier draft | **Withdrawn.** Artefact of whole-image OCR scored against per-sentence references (pitfall (i), §RQ1 (measurement)). |
| DA F₀.₅ = 45.97% | earlier draft | **Withdrawn.** Produced while the domain-adaptation data was binarised against a WordPiece dictionary without tokenisation, sending 91.6% of source tokens to `<unk>`. `prepare_real_da.py` now aborts if OOV ≥ 5%. |

The measurement bugs behind these are described in §RQ1 (measurement) of the paper because they are
easy to repeat and cost us months.
