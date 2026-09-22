# Results manifest

Provenance for every prediction file. All CSVs are one row per test sentence, in a fixed
order shared across files — row *i* is the same sentence everywhere, which is what makes
the paired tests valid.

| File | Rows | Produced by | Feeds |
|---|---|---|---|
| `e2e_ocr_outputs.csv` | 201 | PP-OCRv4 over region crops of the ITIEC-Real test split | every "OCR only" number |
| `e2e_C3_nm2.csv` | 201 | `gen_eval_da.py` on the noise-matched corrector, over L2-normalised OCR | Tab. ladder, Tab. sig; Fig. gating |
| `ocr_confusion_fig.json` | — | `build_ocr_confusion.py` over train+val crops | the OCR noise channel (Eq. channel) |
| `demo_pred.csv` | — | small worked example | `L2_normalizer.py` demo |

## Column schema

`e2e_ocr_outputs.csv`

| Column | Meaning |
|---|---|
| `crop` | path to the region crop |
| `category` | dominant error category (EJA/TIP/ALG/SEM/MOR/SIN/MIX) |
| `teks_ocr` | **human transcription** — the oracle input, not OCR output |
| `paddleocr_out` | PP-OCRv4 hypothesis — the real recognition output |
| `kalimat_koreksi` | gold corrected sentence (the reference) |

`e2e_C3_nm2.csv` and all baseline CSVs

| Column | Meaning |
|---|---|
| `id` | row index; matches position in `e2e_ocr_outputs.csv` |
| `category` | as above |
| `source` | corrector input (for C3: the L2-normalised OCR text) |
| `hyp` | system output |
| `gec_score` | model confidence, used by the gate (C3 only) |
| `kalimat_koreksi` | gold reference |

## Determinism

- `paired_test.py`: paired bootstrap, 5,000 resamples, `seed=1`.
- `eval_protocol_v2.py`: bootstrap CI, 1,000 resamples.
- `knowledge_curve.py`: 5 random seeds per subsampling point, reported as mean ± s.d.
- `prepare_real_da.py` / `gen_noisy_data.py`: `random.seed(42)`.

## The two files the central claim rests on

`e2e_ocr_outputs.csv` and `e2e_C3_nm2.csv`. Given only those two and the Python standard
library, `paired_test.py` reproduces Tab. ladder and Tab. sig — the deterministic normaliser
winning 45–0 over raw OCR, and the neural corrector's 39–38 tie against the normaliser.
Everything else in the paper is context around that comparison.
