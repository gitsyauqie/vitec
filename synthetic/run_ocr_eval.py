"""
run_ocr_eval.py — E1: Domain Gap Measurement via PaddleOCR PP-OCRv4.

Runs PaddleOCR text recognition on synthetic (or real) ITIEC images,
compares OCR output against the reference text, and computes:
  - CER  (Character Error Rate)   — main metric
  - WER  (Word Error Rate)
  - Exact Match Rate

Results are broken down per error category, enabling:
  1. Baseline OCR accuracy on synthetic images (run now, before GEC)
  2. Domain gap measurement when re-run on real annotated images

Usage
-----
# Install PaddleOCR first (run once on server):
#   pip install paddlepaddle-gpu paddleocr --break-system-packages
#   (CPU-only: pip install paddlepaddle paddleocr --break-system-packages)

# Evaluate on synthetic data (sample 200 per category):
python run_ocr_eval.py \\
    --metadata /root/autodl-tmp/paper5/ITIEC_Synthetic/ITIEC_Syn_50K/metadata_clean.csv \\
    --images   /root/autodl-tmp/paper5/ITIEC_Synthetic/ITIEC_Syn_50K/images \\
    --sample   200

# Evaluate on real annotated images (when available):
python run_ocr_eval.py \\
    --metadata /root/autodl-tmp/paper5/ITIEC_Real/metadata_real.csv \\
    --images   /root/autodl-tmp/paper5/ITIEC_Real/images \\
    --sample   0   # 0 = use all rows

Output
------
  ocr_eval_results.csv     — per-image OCR output + CER
  ocr_eval_report.txt      — aggregated statistics
"""

import os
import csv
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ── Levenshtein / CER / WER ───────────────────────────────────────────────────

def _edit_distance(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance (char-level)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # Use two rows to save memory
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1,       # deletion
                          curr[j - 1] + 1,   # insertion
                          prev[j - 1] + cost) # substitution
        prev, curr = curr, prev
    return prev[lb]


def cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate = edit_distance(hyp, ref) / len(ref).
    Returns 0.0 if reference is empty.
    Both strings are normalized (lowercased, stripped).
    """
    hyp = hypothesis.strip().lower()
    ref = reference.strip().lower()
    if len(ref) == 0:
        return 0.0
    return _edit_distance(hyp, ref) / len(ref)


def wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate = edit_distance_words(hyp, ref) / len(ref_words).
    """
    hyp_words = hypothesis.strip().lower().split()
    ref_words = reference.strip().lower().split()
    if len(ref_words) == 0:
        return 0.0
    return _edit_distance(hyp_words, ref_words) / len(ref_words)


def exact_match(hypothesis: str, reference: str) -> bool:
    return hypothesis.strip().lower() == reference.strip().lower()


# ── PaddleOCR wrapper ─────────────────────────────────────────────────────────

def load_ocr_engine(engine: str = "easyocr", use_gpu: bool = True,
                    lang: str = "en", model_size: str = "mobile"):
    """
    Load OCR engine. Supports 'easyocr', 'paddleocr', and 'tesseract'.

    EasyOCR is recommended: PyTorch-based, supports all CUDA versions
    including CUDA 13.x (RTX 5090). PaddleOCR requires CUDA 11/12.
    Tesseract v5 (LSTM) is CPU-only but widely used as a classic baseline.

    Parameters
    ----------
    engine     : 'easyocr' (recommended), 'paddleocr', or 'tesseract'
    use_gpu    : use GPU if available (not applicable to tesseract)
    lang       : language code
    model_size : 'mobile' or 'server' (PaddleOCR only)
    """
    if engine == "easyocr":
        return _load_easyocr(use_gpu)
    elif engine == "tesseract":
        return _load_tesseract()
    else:
        return _load_paddleocr(use_gpu, lang, model_size)


def _load_easyocr(use_gpu: bool = True):
    """Load EasyOCR — PyTorch-based, works with any CUDA version."""
    try:
        import easyocr
    except ImportError:
        print("ERROR: EasyOCR not installed.")
        print("Run: pip install easyocr --break-system-packages")
        sys.exit(1)

    print(f"Loading EasyOCR (gpu={use_gpu}, langs=['en','id']) ...", flush=True)
    # 'en' covers Latin characters used in Indonesian
    # Adding 'id' if available improves Indonesian word recognition
    try:
        reader = easyocr.Reader(['en', 'id'], gpu=use_gpu, verbose=False)
    except Exception:
        reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)

    # Wrap in a simple object with consistent interface
    class EasyOCRWrapper:
        def __init__(self, r):
            self._reader = r
            self.engine = "easyocr"
        def predict(self, image_path):
            return self._reader.readtext(image_path, detail=1)

    print("EasyOCR loaded.\n")
    return EasyOCRWrapper(reader)


def _load_tesseract():
    """
    Load Tesseract v5 (LSTM engine) via pytesseract.

    Requirements (install once on server):
      sudo apt-get install -y tesseract-ocr tesseract-ocr-ind
      pip install pytesseract Pillow --break-system-packages

    PSM 6  = uniform block of text (good for scene text images)
    OEM 1  = LSTM engine only (Tesseract v5)
    Lang   = 'eng+ind' if tesseract-ocr-ind installed, else 'eng'
    """
    try:
        import pytesseract
        from PIL import Image as _PILImage
    except ImportError:
        print("ERROR: pytesseract / Pillow not installed.")
        print("Run: pip install pytesseract Pillow --break-system-packages")
        print("Also: sudo apt-get install -y tesseract-ocr tesseract-ocr-ind")
        sys.exit(1)

    # Check tesseract binary is available
    try:
        ver = pytesseract.get_tesseract_version()
        print(f"Tesseract version : {ver}")
    except pytesseract.TesseractNotFoundError:
        print("ERROR: Tesseract binary not found. Install with:")
        print("  sudo apt-get install -y tesseract-ocr tesseract-ocr-ind")
        sys.exit(1)

    # Determine available languages
    available_langs = pytesseract.get_languages(config="")
    if "ind" in available_langs:
        lang_str = "eng+ind"
    else:
        lang_str = "eng"
        print("WARNING: 'ind' language pack not installed. Using 'eng' only.")
        print("  Install: sudo apt-get install -y tesseract-ocr-ind")

    tess_config = f"--oem 1 --psm 6"
    print(f"Tesseract loaded  : lang={lang_str}, config='{tess_config}'\n")

    class TesseractWrapper:
        def __init__(self, lang, config):
            self._lang = lang
            self._config = config
            self.engine = "tesseract"

        def predict(self, image_path):
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            text = pytesseract.image_to_string(img, lang=self._lang,
                                               config=self._config)
            return text.strip()

    return TesseractWrapper(lang_str, tess_config)


def _load_paddleocr(use_gpu: bool = True, lang: str = "en", model_size: str = "mobile"):
    """
    Load PaddleOCR (requires PaddlePaddle compiled for matching CUDA version).
    NOTE: As of 2026, PaddlePaddle does not support CUDA 13.x (RTX 5090).
          Use EasyOCR instead in that environment.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        print("ERROR: PaddleOCR not installed.")
        print("Run: pip install paddlepaddle-gpu paddleocr --break-system-packages")
        sys.exit(1)

    try:
        import paddleocr as _poc
        version_str = getattr(_poc, "__version__", "0.0.0")
        parts = version_str.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        is_new_api = (major > 2) or (major == 2 and minor >= 8)
    except Exception:
        is_new_api = False

    device_str = "gpu" if use_gpu else "cpu"
    print(f"Loading PaddleOCR (device={device_str}, lang={lang}, "
          f"size={model_size}, new_api={is_new_api}) ...", flush=True)

    if is_new_api:
        det_model = "PP-OCRv4_mobile_det" if model_size == "mobile" else "PP-OCRv5_server_det"
        rec_model = "en_PP-OCRv4_mobile_rec" if model_size == "mobile" else "en_PP-OCRv5_server_rec"
        try:
            ocr = PaddleOCR(use_textline_orientation=True, lang=lang,
                            device=device_str,
                            text_detection_model_name=det_model,
                            text_recognition_model_name=rec_model)
        except Exception:
            try:
                ocr = PaddleOCR(use_textline_orientation=True, lang=lang, device=device_str)
            except Exception:
                ocr = PaddleOCR(lang=lang)
    else:
        try:
            ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        except Exception:
            ocr = PaddleOCR(lang=lang)

    # Wrap for consistent interface
    class PaddleOCRWrapper:
        def __init__(self, o):
            self._ocr = o
            self.engine = "paddleocr"
        def predict(self, image_path):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(self._ocr, "predict"):
                    return self._ocr.predict(image_path)
                return self._ocr.ocr(image_path)

    print("PaddleOCR loaded.\n")
    return PaddleOCRWrapper(ocr)


def run_ocr(ocr, image_path: str) -> str:
    """
    Run OCR engine on a single image, return concatenated text.
    Handles both EasyOCR and PaddleOCR wrapper interfaces.
    """
    try:
        engine = getattr(ocr, "engine", "unknown")
        result = ocr.predict(image_path)

        if result is None:
            return ""

        lines = []

        # ── Tesseract format: returns plain string directly ───────────────────
        if engine == "tesseract":
            return result if isinstance(result, str) else ""

        # ── EasyOCR format: list of (bbox, text, conf) ────────────────────────
        if engine == "easyocr":
            for item in result:
                if item is None:
                    continue
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    # item = (bbox, text) or (bbox, text, conf)
                    text = item[1] if len(item) >= 2 else ""
                    if text and isinstance(text, str):
                        lines.append(text)

        # ── PaddleOCR format: various depending on version ────────────────────
        else:
            if isinstance(result, (list, tuple)):
                for item in result:
                    if item is None:
                        continue
                    # New API: result object with rec_texts
                    rec_texts = getattr(item, "rec_texts", None)
                    if rec_texts is not None:
                        lines.extend([str(t) for t in rec_texts if t])
                        continue
                    # Dict format
                    if isinstance(item, dict):
                        texts = item.get("rec_texts", []) or []
                        lines.extend([str(t) for t in texts if t])
                        continue
                    # Old nested list: [[bbox], (text, conf)]
                    if isinstance(item, (list, tuple)):
                        for line in item:
                            if isinstance(line, (list, tuple)) and len(line) == 2:
                                text_conf = line[1]
                                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                                    text = text_conf[0]
                                    if text and isinstance(text, str):
                                        lines.append(text)

        return " ".join(lines).strip()
    except Exception:
        return ""


# ── Sampling ──────────────────────────────────────────────────────────────────

ALL_CATEGORIES = [
    "morphological", "syntactic", "semantic",
    "spelling", "typographic", "algospeak",
]


def sample_rows(metadata_path: str, n_per_cat: int, seed: int, images_dir: str):
    """
    Load metadata.csv and return a stratified random sample.
    n_per_cat=0 means use ALL rows for each category.
    Only includes rows whose image file actually exists.
    """
    by_cat = defaultdict(list)

    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Resolve image path
            img_path = row.get("image_path", "")
            if not os.path.isabs(img_path):
                img_path = os.path.join(images_dir, os.path.basename(img_path))
            row["_resolved_path"] = img_path

            if os.path.exists(img_path):
                by_cat[row["category"]].append(row)

    rng = random.Random(seed)
    sampled = []
    for cat in ALL_CATEGORIES:
        rows = by_cat.get(cat, [])
        if n_per_cat == 0 or len(rows) <= n_per_cat:
            chosen = rows
        else:
            chosen = rng.sample(rows, n_per_cat)
        sampled.extend(chosen)
        print(f"  {cat:<16}: {len(chosen):>5,} images selected (available: {len(rows):,})")

    return sampled


# ── Debug helper ─────────────────────────────────────────────────────────────

def _debug_ocr_output(ocr, image_path: str):
    """Print raw OCR output for one image to confirm parsing works."""
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Engine: {getattr(ocr, 'engine', 'unknown')}")
    try:
        raw = ocr.predict(image_path)
        raw_list = list(raw) if hasattr(raw, "__next__") else raw
        print(f"  Raw type   : {type(raw_list)}")
        if raw_list:
            print(f"  First item : {repr(raw_list[0])[:150]}")
    except Exception as e:
        print(f"  predict() error: {e}")
    extracted = run_ocr(ocr, image_path)
    print(f"  Extracted  : '{extracted[:120]}'")
    print()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(ocr, rows, out_dir: str, reference_col: str = "src_text"):
    """
    Run OCR on all rows, compute metrics, return results list.

    reference_col: which metadata column to use as OCR reference.
      'src_text' for synthetic (OCR should reproduce the erroneous text)
      'trg_text' for real annotated images (ground-truth corrected text)
    """
    results = []
    total = len(rows)

    print(f"\nRunning OCR on {total:,} images (reference: '{reference_col}') ...")
    print(f"{'─'*60}")

    for i, row in enumerate(rows, 1):
        img_path  = row["_resolved_path"]
        reference = row.get(reference_col, "").strip()
        category  = row.get("category", "unknown")

        ocr_text = run_ocr(ocr, img_path)

        c = cer(ocr_text, reference)
        w = wer(ocr_text, reference)
        em = exact_match(ocr_text, reference)

        results.append({
            "image_id":    row.get("image_id", ""),
            "image_path":  img_path,
            "category":    category,
            "reference":   reference,
            "ocr_output":  ocr_text,
            "cer":         round(c, 4),
            "wer":         round(w, 4),
            "exact_match": int(em),
        })

        # Progress every 50 images
        if i % 50 == 0 or i == total:
            avg_cer = sum(r["cer"] for r in results) / len(results)
            print(f"  [{i:>5}/{total}]  avg CER so far: {avg_cer:.4f}")

    return results


# ── Aggregation & report ──────────────────────────────────────────────────────

def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _pct(n, total):
    return (n / total * 100) if total > 0 else 0.0


def build_report(results, reference_col, args):
    """Build aggregated report lines."""
    lines = []
    lines.append("=" * 65)
    lines.append(" ITIEC — OCR Baseline Evaluation Report  (E1: Domain Gap)")
    lines.append(f" Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f" Reference : {reference_col}")
    engine_name = getattr(args, "engine", "easyocr").upper()
    lines.append(f" Model     : {engine_name}  (lang=en/id)")
    lines.append(f" Images    : {args.images}")
    lines.append(f" Metadata  : {args.metadata}")
    lines.append("=" * 65)

    total = len(results)
    global_cer = _mean([r["cer"] for r in results])
    global_wer = _mean([r["wer"] for r in results])
    global_em  = _mean([r["exact_match"] for r in results]) * 100

    lines.append(f"\n{'─'*65}")
    lines.append("GLOBAL METRICS")
    lines.append(f"{'─'*65}")
    lines.append(f"  Total images evaluated : {total:,}")
    lines.append(f"  Mean CER               : {global_cer:.4f}  ({global_cer*100:.2f}%)")
    lines.append(f"  Mean WER               : {global_wer:.4f}  ({global_wer*100:.2f}%)")
    lines.append(f"  Exact Match Rate       : {global_em:.2f}%")
    lines.append(f"  OCR Accuracy (1-CER)   : {(1-global_cer)*100:.2f}%")

    # CER distribution buckets
    cer_buckets = {
        "Perfect (CER=0)":    sum(1 for r in results if r["cer"] == 0),
        "Good    (CER≤0.05)": sum(1 for r in results if 0 < r["cer"] <= 0.05),
        "Fair    (CER≤0.20)": sum(1 for r in results if 0.05 < r["cer"] <= 0.20),
        "Poor    (CER≤0.50)": sum(1 for r in results if 0.20 < r["cer"] <= 0.50),
        "Failed  (CER>0.50)": sum(1 for r in results if r["cer"] > 0.50),
    }
    lines.append(f"\n  CER distribution:")
    for label, n in cer_buckets.items():
        bar = "█" * int(_pct(n, total) / 2)
        lines.append(f"    {label}  {n:>6,}  ({_pct(n, total):>5.1f}%)  {bar}")

    # Per-category breakdown
    lines.append(f"\n{'─'*65}")
    lines.append("PER-CATEGORY METRICS")
    lines.append(f"{'─'*65}")
    lines.append(
        f"  {'Category':<16}  {'N':>6}  {'CER':>7}  {'WER':>7}  "
        f"{'ExactMatch':>11}  {'OCRAcc':>8}"
    )
    lines.append(f"  {'─'*14}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*11}  {'─'*8}")

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    cat_stats = {}
    for cat in ALL_CATEGORIES:
        cat_rows = by_cat.get(cat, [])
        if not cat_rows:
            continue
        c = _mean([r["cer"] for r in cat_rows])
        w = _mean([r["wer"] for r in cat_rows])
        em = _mean([r["exact_match"] for r in cat_rows]) * 100
        acc = (1 - c) * 100
        cat_stats[cat] = {"cer": c, "wer": w, "em": em, "acc": acc, "n": len(cat_rows)}
        lines.append(
            f"  {cat:<16}  {len(cat_rows):>6,}  {c:>7.4f}  {w:>7.4f}  "
            f"{em:>10.2f}%  {acc:>7.2f}%"
        )

    # Worst & best categories
    if cat_stats:
        best = min(cat_stats.items(), key=lambda x: x[1]["cer"])
        worst = max(cat_stats.items(), key=lambda x: x[1]["cer"])
        lines.append(f"\n  Best  category: {best[0]}  (CER={best[1]['cer']:.4f})")
        lines.append(f"  Worst category: {worst[0]}  (CER={worst[1]['cer']:.4f})")

    # OCR failure analysis — show worst 10 examples
    lines.append(f"\n{'─'*65}")
    lines.append("WORST 10 OCR OUTPUTS (highest CER)")
    lines.append(f"{'─'*65}")
    worst10 = sorted(results, key=lambda r: r["cer"], reverse=True)[:10]
    for i, r in enumerate(worst10, 1):
        ref_preview = r["reference"][:60] + "..." if len(r["reference"]) > 60 else r["reference"]
        ocr_preview = r["ocr_output"][:60] + "..." if len(r["ocr_output"]) > 60 else r["ocr_output"]
        lines.append(f"\n  [{i}] {os.path.basename(r['image_path'])}  cat={r['category']}  CER={r['cer']:.4f}")
        lines.append(f"      REF: {ref_preview}")
        lines.append(f"      OCR: {ocr_preview}")

    # Paper-ready table (LaTeX snippet)
    lines.append(f"\n{'─'*65}")
    lines.append("LATEX TABLE SNIPPET (copy to paper5_IEEE.tex)")
    lines.append(f"{'─'*65}")
    lines.append(r"  \begin{tabular}{lrrr}")
    lines.append(r"  \toprule")
    lines.append(r"  Category & CER $\downarrow$ & WER $\downarrow$ & EM (\%) $\uparrow$ \\")
    lines.append(r"  \midrule")
    for cat in ALL_CATEGORIES:
        if cat not in cat_stats:
            continue
        s = cat_stats[cat]
        lines.append(
            f"  {cat.capitalize()} & "
            f"{s['cer']:.4f} & {s['wer']:.4f} & {s['em']:.1f} \\\\"
        )
    lines.append(r"  \midrule")
    lines.append(
        f"  Overall & {global_cer:.4f} & {global_wer:.4f} & {global_em:.1f} \\\\"
    )
    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")

    lines.append(f"\n{'='*65}")
    lines.append(" End of Report")
    lines.append(f"{'='*65}")

    return lines, cat_stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="E1: OCR baseline evaluation using PaddleOCR PP-OCRv4."
    )
    parser.add_argument("--metadata", required=True,
        help="Path to metadata_clean.csv (or real metadata CSV).")
    parser.add_argument("--images", required=True,
        help="Path to images directory.")
    parser.add_argument("--out", default=None,
        help="Output directory (default: same folder as metadata).")
    parser.add_argument("--sample", type=int, default=200,
        help="Images per category to evaluate (0 = all, default: 200).")
    parser.add_argument("--reference", default="src_text",
        choices=["src_text", "trg_text"],
        help="Which column to use as OCR reference text. "
             "Use 'src_text' for synthetic (OCR target is the erroneous text). "
             "Use 'trg_text' for real annotated images (ground-truth). "
             "(default: src_text)")
    parser.add_argument("--cpu", action="store_true",
        help="Force CPU mode (slower, but works without GPU).")
    parser.add_argument("--engine", default="easyocr",
        choices=["easyocr", "paddleocr", "tesseract"],
        help="OCR engine to use. 'easyocr' (default) = PyTorch-based, works with "
             "all CUDA versions including 13.x (RTX 5090). "
             "'paddleocr' = requires CUDA 11/12. "
             "'tesseract' = Tesseract v5 LSTM (CPU-only classic baseline).")
    parser.add_argument("--model-size", default="mobile",
        choices=["mobile", "server"],
        help="Model size for PaddleOCR only. Default: mobile.")
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed for sampling (default: 42).")
    parser.add_argument("--lang", default="en",
        help="PaddleOCR language model (default: 'en' — covers Latin/Indonesian).")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.metadata))
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f" ITIEC OCR Evaluation — E1: Domain Gap Measurement")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f" metadata  : {args.metadata}")
    print(f" images    : {args.images}")
    print(f" sample    : {args.sample if args.sample > 0 else 'ALL'} per category")
    print(f" reference : {args.reference}")
    print(f" gpu       : {not args.cpu}")
    print(f"{'='*65}\n")

    # ── Sample rows ───────────────────────────────────────────────────────────
    print("Sampling images ...")
    rows = sample_rows(
        metadata_path=args.metadata,
        n_per_cat=args.sample,
        seed=args.seed,
        images_dir=args.images,
    )
    print(f"\nTotal selected: {len(rows):,} images\n")

    if len(rows) == 0:
        print("ERROR: No valid images found. Check --metadata and --images paths.")
        sys.exit(1)

    # ── Load OCR model ────────────────────────────────────────────────────────
    use_gpu = not args.cpu
    engine = getattr(args, "engine", "easyocr")
    model_size = getattr(args, "model_size", "mobile").replace("-", "_")
    ocr = load_ocr_engine(engine=engine, use_gpu=use_gpu,
                          lang=args.lang, model_size=model_size)

    # ── Sanity check: test OCR on first image, print raw output ──────────────
    print("Sanity check on first image ...")
    first_img = rows[0]["_resolved_path"]
    _debug_ocr_output(ocr, first_img)

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = evaluate(ocr, rows, out_dir, reference_col=args.reference)

    # ── Build report ──────────────────────────────────────────────────────────
    report_lines, cat_stats = build_report(results, args.reference, args)

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    for line in report_lines:
        print(line)

    # ── Save results CSV ──────────────────────────────────────────────────────
    results_csv = os.path.join(out_dir, "ocr_eval_results.csv")
    report_txt  = os.path.join(out_dir, "ocr_eval_report.txt")
    summary_json = os.path.join(out_dir, "ocr_eval_summary.json")

    with open(results_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "image_path", "category",
                        "reference", "ocr_output", "cer", "wer", "exact_match"]
        )
        writer.writeheader()
        writer.writerows(results)

    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # Save summary as JSON for easy programmatic use later
    global_cer = _mean([r["cer"] for r in results])
    global_wer = _mean([r["wer"] for r in results])
    summary = {
        "timestamp":    datetime.now().isoformat(),
        "model":        "PaddleOCR-PP-OCRv4",
        "reference_col": args.reference,
        "n_evaluated":  len(results),
        "global_cer":   round(global_cer, 4),
        "global_wer":   round(global_wer, 4),
        "global_em_pct": round(_mean([r["exact_match"] for r in results]) * 100, 2),
        "per_category": {
            cat: {k: round(v, 4) for k, v in s.items()}
            for cat, s in cat_stats.items()
        }
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅  Results CSV  → {results_csv}")
    print(f"✅  Report       → {report_txt}")
    print(f"✅  Summary JSON → {summary_json}")
    print(f"\nNext step:")
    print(f"  • When real annotated images are ready, re-run with --reference trg_text")
    print(f"  • Compare global_cer synthetic vs real → domain gap value for paper")


if __name__ == "__main__":
    main()
