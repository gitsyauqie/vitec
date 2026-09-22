"""
verify_synthetic.py — Quality verification script for ITIEC-Syn dataset.

Checks the generated synthetic dataset (metadata.csv + images/) for:
  1. Row count and completeness
  2. Category distribution vs expected weights
  3. Text length statistics (src vs trg, per category)
  4. Train/val/test split distribution
  5. Image file integrity (readable, non-zero, size distribution)
  6. Sample src/trg pair inspection (error injection quality)
  7. Augmentation type distribution
  8. Duplicate detection (src text)

Produces:
  - Console report (stdout)
  - verification_report.txt  (same content, saved alongside metadata.csv)
  - sample_pairs.txt         (random 30 src/trg pairs per category for manual review)

Usage
-----
Run on the AutoDL server where the dataset lives:

    python verify_synthetic.py --out ./output

Or point to a specific metadata file:

    python verify_synthetic.py --metadata ./output/metadata.csv --images ./output/images

Author: ITIEC research pipeline
"""

import os
import sys
import csv
import argparse
import random
import math
import io
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime


# ── Expected configuration (must match generate_synthetic.py) ─────────────────

EXPECTED_WEIGHTS = {
    "morphological":  0.25,
    "syntactic":      0.25,
    "semantic":       0.15,
    "spelling":       0.15,
    "typographic":    0.10,
    "algospeak":      0.10,
}

EXPECTED_SPLITS = {
    "train": 0.80,
    "val":   0.10,
    "test":  0.10,
}

ALL_CATEGORIES = list(EXPECTED_WEIGHTS.keys())

# Tolerance for distribution checks (±5 percentage points)
DIST_TOLERANCE = 0.05


# ── Utilities ─────────────────────────────────────────────────────────────────

def _pct(n, total):
    return (n / total * 100) if total > 0 else 0.0


def _bar(value, total, width=30):
    """Simple ASCII progress bar."""
    filled = int(round(value / total * width)) if total > 0 else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _median(values):
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _pct_in_range(values, lo, hi):
    """Percentage of values in [lo, hi]."""
    if not values:
        return 0.0
    return sum(1 for v in values if lo <= v <= hi) / len(values) * 100


# ── Core verification functions ───────────────────────────────────────────────

def load_metadata(metadata_path: str):
    """
    Stream-read metadata.csv without loading everything into RAM.
    Returns (rows list, fieldnames list).
    Rows are dicts.
    """
    rows = []
    fieldnames = None
    with open(metadata_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def check_schema(fieldnames, out):
    """Verify expected columns are present."""
    # Accept either 'augmentation' or 'augmentation_intensity'
    core_cols = {"image_id", "image_path", "src_text", "trg_text", "category", "split"}
    aug_variants = {"augmentation", "augmentation_intensity"}

    present = set(fieldnames or [])
    missing = core_cols - present
    has_aug = bool(aug_variants & present)

    if not has_aug:
        missing.add("augmentation / augmentation_intensity")

    extra = present - core_cols - aug_variants
    if missing:
        out.append(f"  ⚠️  MISSING columns: {sorted(missing)}")
        return False
    if extra:
        out.append(f"  ℹ️  Extra columns (OK): {sorted(extra)}")
    aug_found = aug_variants & present
    out.append(f"  ✅  All expected columns present (aug col: {aug_found})")
    return True


def check_category_distribution(rows, out, samples_out):
    """Check category counts vs expected weights; collect samples."""
    cat_counts = Counter(r["category"] for r in rows)
    total = len(rows)

    out.append(f"\n{'─'*60}")
    out.append("2. CATEGORY DISTRIBUTION")
    out.append(f"{'─'*60}")
    out.append(f"  {'Category':<16} {'Count':>7}  {'Actual%':>8}  {'Expected%':>10}  {'Δ':>7}  {'Bar'}")

    ok = True
    samples = defaultdict(list)
    for r in rows:
        samples[r["category"]].append(r)

    for cat in ALL_CATEGORIES:
        n = cat_counts.get(cat, 0)
        actual_frac  = n / total if total else 0
        expected_frac = EXPECTED_WEIGHTS.get(cat, 0)
        delta = actual_frac - expected_frac
        flag = "⚠️" if abs(delta) > DIST_TOLERANCE else "✅"
        bar = _bar(n, total)
        out.append(
            f"  {flag} {cat:<14} {n:>7,}  {actual_frac*100:>7.1f}%  "
            f"{expected_frac*100:>8.1f}%  {delta*100:>+6.1f}%  {bar}"
        )
        if abs(delta) > DIST_TOLERANCE:
            ok = False

    unknown = set(cat_counts.keys()) - set(ALL_CATEGORIES)
    if unknown:
        out.append(f"\n  ⚠️  Unknown categories found: {unknown}")
        ok = False

    out.append(f"\n  Total rows: {total:,}")
    out.append(f"  Distribution {'OK ✅' if ok else 'DEVIATION FOUND ⚠️'} (tolerance ±{DIST_TOLERANCE*100:.0f}pp)")

    # Build sample pairs per category (up to 5 each for report)
    rng = random.Random(42)
    for cat in ALL_CATEGORIES:
        cat_rows = samples.get(cat, [])
        chosen = rng.sample(cat_rows, min(5, len(cat_rows)))
        samples_out[cat].extend(chosen)

    return cat_counts, ok


def check_text_lengths(rows, cat_counts, out):
    """Text length stats: src vs trg, per category."""
    out.append(f"\n{'─'*60}")
    out.append("3. TEXT LENGTH STATISTICS")
    out.append(f"{'─'*60}")

    # Global stats
    src_lens = [len(r["src_text"]) for r in rows if r.get("src_text")]
    trg_lens = [len(r["trg_text"]) for r in rows if r.get("trg_text")]
    empty_src = sum(1 for r in rows if not r.get("src_text", "").strip())
    empty_trg = sum(1 for r in rows if not r.get("trg_text", "").strip())

    out.append(f"\n  GLOBAL (chars)   {'src':>8}   {'trg':>8}")
    out.append(f"  {'Mean':<16} {_mean(src_lens):>8.1f}   {_mean(trg_lens):>8.1f}")
    out.append(f"  {'Median':<16} {_median(src_lens):>8.1f}   {_median(trg_lens):>8.1f}")
    out.append(f"  {'Min':<16} {min(src_lens, default=0):>8}   {min(trg_lens, default=0):>8}")
    out.append(f"  {'Max':<16} {max(src_lens, default=0):>8}   {max(trg_lens, default=0):>8}")
    out.append(f"  {'Empty':<16} {empty_src:>8,}   {empty_trg:>8,}")

    if empty_src > 0 or empty_trg > 0:
        out.append(f"  ⚠️  Empty texts found! src={empty_src}, trg={empty_trg}")
    else:
        out.append(f"  ✅  No empty texts")

    # Per-category
    out.append(f"\n  PER CATEGORY (src chars):")
    out.append(f"  {'Category':<16} {'N':>7}  {'Mean':>7}  {'Median':>7}  {'Min':>5}  {'Max':>5}  {'≤5 chars%':>10}")

    by_cat = defaultdict(list)
    for r in rows:
        if r.get("src_text"):
            by_cat[r["category"]].append(len(r["src_text"]))

    issues = False
    for cat in ALL_CATEGORIES:
        lens = by_cat.get(cat, [])
        if not lens:
            out.append(f"  ⚠️  {cat:<14}  NO DATA")
            issues = True
            continue
        short_pct = _pct_in_range(lens, 0, 5)
        flag = "⚠️" if short_pct > 10 else "  "
        out.append(
            f"  {flag} {cat:<14} {len(lens):>7,}  {_mean(lens):>7.1f}  "
            f"{_median(lens):>7.1f}  {min(lens):>5}  {max(lens):>5}  {short_pct:>9.1f}%"
        )
        if short_pct > 10:
            issues = True

    out.append(f"\n  Text lengths {'OK ✅' if not issues else 'ISSUES FOUND ⚠️'}")
    return not issues


def check_splits(rows, out):
    """Verify train/val/test split distribution."""
    out.append(f"\n{'─'*60}")
    out.append("4. TRAIN/VAL/TEST SPLIT")
    out.append(f"{'─'*60}")

    split_counts = Counter(r.get("split", "unknown") for r in rows)
    total = len(rows)
    ok = True

    for split, expected in EXPECTED_SPLITS.items():
        n = split_counts.get(split, 0)
        actual = n / total if total else 0
        delta = actual - expected
        flag = "⚠️" if abs(delta) > DIST_TOLERANCE else "✅"
        if abs(delta) > DIST_TOLERANCE:
            ok = False
        out.append(
            f"  {flag} {split:<8} {n:>7,}  {actual*100:>7.1f}%  "
            f"(expected {expected*100:.0f}%  Δ{delta*100:+.1f}%)"
        )

    unknown_splits = set(split_counts.keys()) - set(EXPECTED_SPLITS.keys())
    if unknown_splits:
        out.append(f"  ⚠️  Unknown split values: {unknown_splits}")
        ok = False

    out.append(f"\n  Split distribution {'OK ✅' if ok else 'DEVIATION ⚠️'}")
    return ok


def check_augmentation(rows, out):
    """Augmentation type distribution check."""
    out.append(f"\n{'─'*60}")
    out.append("5. AUGMENTATION DISTRIBUTION")
    out.append(f"{'─'*60}")

    # Column may be named 'augmentation' or 'augmentation_intensity'
    aug_col = "augmentation"
    if rows and "augmentation_intensity" in rows[0] and "augmentation" not in rows[0]:
        aug_col = "augmentation_intensity"
    out.append(f"  (column: '{aug_col}')")

    aug_counts = Counter(r.get(aug_col, "unknown") or "unknown" for r in rows)
    total = len(rows)

    for aug, n in sorted(aug_counts.items(), key=lambda x: -x[1]):
        bar = _bar(n, total, width=20)
        out.append(f"  {aug:<12} {n:>7,}  {_pct(n, total):>6.1f}%  {bar}")

    missing_aug = sum(1 for r in rows if not (r.get(aug_col) or "").strip())
    if missing_aug:
        out.append(f"  ⚠️  {missing_aug:,} rows with missing augmentation field")
    else:
        out.append(f"  ✅  All rows have augmentation field")

    return True


def check_image_integrity(rows, images_dir, out, max_check=5000):
    """
    Check image files exist and are readable.
    Checks up to max_check files (random sample if more).
    """
    out.append(f"\n{'─'*60}")
    out.append("6. IMAGE FILE INTEGRITY")
    out.append(f"{'─'*60}")

    # Try to import PIL for image validation
    try:
        from PIL import Image
        pil_available = True
    except ImportError:
        pil_available = False
        out.append("  ℹ️  Pillow not available — checking file existence only")

    total = len(rows)
    rng = random.Random(42)

    # Sample rows to check
    check_rows = rows if total <= max_check else rng.sample(rows, max_check)
    out.append(f"  Checking {len(check_rows):,} / {total:,} images "
               f"({'full' if total <= max_check else 'random sample'})")

    missing    = 0
    unreadable = 0
    zero_size  = 0
    sizes      = []
    widths     = []
    heights    = []

    for row in check_rows:
        img_path = row.get("image_path", "")
        # image_path in CSV may be relative — resolve against images_dir
        if not os.path.isabs(img_path):
            img_path = os.path.join(images_dir, os.path.basename(img_path))

        if not os.path.exists(img_path):
            missing += 1
            continue

        sz = os.path.getsize(img_path)
        if sz == 0:
            zero_size += 1
            continue
        sizes.append(sz)

        if pil_available:
            try:
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except Exception:
                unreadable += 1

    checked = len(check_rows)
    ok_count = checked - missing - zero_size - unreadable
    ok = (missing + zero_size + unreadable) == 0

    out.append(f"\n  {'✅' if missing == 0 else '⚠️'}  Missing files  : {missing:,} / {checked:,}")
    out.append(f"  {'✅' if zero_size == 0 else '⚠️'}  Zero-size files: {zero_size:,} / {checked:,}")
    if pil_available:
        out.append(f"  {'✅' if unreadable == 0 else '⚠️'}  Unreadable PNG : {unreadable:,} / {checked:,}")
    out.append(f"  ✅  Readable OK   : {ok_count:,} / {checked:,}")

    if sizes:
        avg_kb = _mean(sizes) / 1024
        min_kb = min(sizes) / 1024
        max_kb = max(sizes) / 1024
        out.append(f"\n  File size (KB):  mean={avg_kb:.1f}  min={min_kb:.1f}  max={max_kb:.1f}")

    if widths and heights:
        out.append(f"  Image width  px: mean={_mean(widths):.0f}  min={min(widths)}  max={max(widths)}")
        out.append(f"  Image height px: mean={_mean(heights):.0f}  min={min(heights)}  max={max(heights)}")

    out.append(f"\n  Image integrity {'OK ✅' if ok else 'ISSUES FOUND ⚠️'}")

    # Estimate total dataset size
    if sizes:
        est_total_mb = _mean(sizes) * total / (1024 * 1024)
        out.append(f"  Estimated total image storage: ~{est_total_mb:.0f} MB ({est_total_mb/1024:.1f} GB)")

    return ok


def check_duplicates(rows, out):
    """Check for duplicate src_text values."""
    out.append(f"\n{'─'*60}")
    out.append("7. DUPLICATE DETECTION")
    out.append(f"{'─'*60}")

    src_texts = [r.get("src_text", "") for r in rows]
    total = len(src_texts)
    unique = len(set(src_texts))
    dup_count = total - unique
    dup_pct = _pct(dup_count, total)

    flag = "⚠️" if dup_pct > 5 else "✅"
    out.append(f"  {flag} Total rows  : {total:,}")
    out.append(f"  {flag} Unique src  : {unique:,}")
    out.append(f"  {flag} Duplicates  : {dup_count:,} ({dup_pct:.1f}%)")

    if dup_pct <= 1:
        out.append(f"  ✅  Duplicate rate very low (≤1%) — excellent")
    elif dup_pct <= 5:
        out.append(f"  ✅  Duplicate rate acceptable (≤5%)")
    else:
        out.append(f"  ⚠️  High duplicate rate (>{dup_pct:.1f}%) — may affect model generalisation")

    return dup_pct <= 5


def build_sample_pairs(samples_out, all_rows, out, n_per_cat=5):
    """Generate readable sample pairs for manual review."""
    sample_lines = []
    sample_lines.append("=" * 70)
    sample_lines.append("SAMPLE SRC/TRG PAIRS FOR MANUAL INSPECTION")
    sample_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sample_lines.append("=" * 70)

    rng = random.Random(99)

    for cat in ALL_CATEGORIES:
        cat_rows = [r for r in all_rows if r.get("category") == cat]
        chosen = rng.sample(cat_rows, min(n_per_cat, len(cat_rows)))

        sample_lines.append(f"\n{'─'*70}")
        sample_lines.append(f"CATEGORY: {cat.upper()}  (showing {len(chosen)} of {len(cat_rows):,})")
        sample_lines.append(f"{'─'*70}")

        for i, row in enumerate(chosen, 1):
            src = row.get("src_text", "")
            trg = row.get("trg_text", "")
            aug = row.get("augmentation", "")
            img = os.path.basename(row.get("image_path", ""))

            # Character-level diff highlight
            same = src == trg
            src_len = len(src)
            trg_len = len(trg)

            sample_lines.append(f"\n  [{i}] img={img}  aug={aug}")
            sample_lines.append(f"      SRC ({src_len:>3} chars): {src}")
            sample_lines.append(f"      TRG ({trg_len:>3} chars): {trg}")

            if same:
                sample_lines.append(f"      ⚠️  SRC == TRG  (no correction applied!)")
            else:
                # Count differing chars (simple)
                min_len = min(src_len, trg_len)
                diff_chars = sum(1 for a, b in zip(src, trg) if a != b)
                diff_chars += abs(src_len - trg_len)
                sample_lines.append(f"      ✏️  {diff_chars} char-level differences  (len diff: {src_len - trg_len:+d})")

    out.append("\n  (Sample pairs saved to sample_pairs.txt)")
    return sample_lines


def check_identical_pairs(rows, out):
    """Flag rows where src_text == trg_text (no correction applied)."""
    identical = sum(1 for r in rows if r.get("src_text", "") == r.get("trg_text", ""))
    total = len(rows)
    pct = _pct(identical, total)
    flag = "⚠️" if pct > 2 else "✅"
    out.append(f"\n  {flag} Identical src==trg: {identical:,} / {total:,} ({pct:.1f}%)")

    # Per-category breakdown
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat = r.get("category", "unknown")
        by_cat[cat][1] += 1
        if r.get("src_text", "") == r.get("trg_text", ""):
            by_cat[cat][0] += 1

    issues = []
    for cat in ALL_CATEGORIES:
        n_ident, n_total = by_cat.get(cat, [0, 0])
        if n_total > 0 and n_ident / n_total > 0.02:
            issues.append(f"{cat}: {_pct(n_ident, n_total):.1f}%")

    if issues:
        out.append(f"  ⚠️  High identical rates per category: {', '.join(issues)}")

    return pct <= 2


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify ITIEC-Syn synthetic dataset quality."
    )
    parser.add_argument(
        "--out", default="./output",
        help="Output directory (default: ./output). Looks for metadata.csv and images/ inside."
    )
    parser.add_argument(
        "--metadata",
        help="Explicit path to metadata.csv (overrides --out)."
    )
    parser.add_argument(
        "--images",
        help="Explicit path to images directory (overrides --out)."
    )
    parser.add_argument(
        "--max-image-check", type=int, default=5000,
        help="Max number of images to integrity-check (default 5000)."
    )
    parser.add_argument(
        "--sample-n", type=int, default=5,
        help="Number of sample pairs to show per category (default 5)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default 42)."
    )
    args = parser.parse_args()

    # Resolve paths
    out_dir    = args.out
    meta_path  = args.metadata or os.path.join(out_dir, "metadata.csv")
    images_dir = args.images   or os.path.join(out_dir, "images")

    print(f"\n{'='*60}")
    print(f" ITIEC-Syn Dataset Verification")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f" metadata : {meta_path}")
    print(f" images   : {images_dir}")
    print(f"{'='*60}\n")

    # ── Check paths exist ─────────────────────────────────────────────────────
    if not os.path.exists(meta_path):
        print(f"ERROR: metadata.csv not found at {meta_path}")
        print("       Use --out or --metadata to specify the correct path.")
        sys.exit(1)

    if not os.path.isdir(images_dir):
        print(f"WARNING: images directory not found at {images_dir}")
        print("         Image integrity checks will be skipped.")
        skip_images = True
    else:
        skip_images = False

    # ── Collect report lines ──────────────────────────────────────────────────
    report = []
    report.append("=" * 60)
    report.append(" ITIEC-Syn VERIFICATION REPORT")
    report.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f" metadata : {meta_path}")
    report.append(f" images   : {images_dir}")
    report.append("=" * 60)

    samples_per_cat = defaultdict(list)

    # ── 1. Load & schema check ────────────────────────────────────────────────
    report.append(f"\n{'─'*60}")
    report.append("1. METADATA OVERVIEW")
    report.append(f"{'─'*60}")

    print("Loading metadata.csv ...", end=" ", flush=True)
    rows, fieldnames = load_metadata(meta_path)
    print(f"{len(rows):,} rows loaded.")

    report.append(f"  Total rows: {len(rows):,}")
    report.append(f"  Columns: {fieldnames}")
    schema_ok = check_schema(fieldnames, report)

    checks = {}

    # ── 2. Category distribution ──────────────────────────────────────────────
    cat_counts, checks["category_dist"] = check_category_distribution(rows, report, samples_per_cat)

    # ── 3. Text length stats ──────────────────────────────────────────────────
    checks["text_lengths"] = check_text_lengths(rows, cat_counts, report)

    # ── 4. Split distribution ─────────────────────────────────────────────────
    checks["splits"] = check_splits(rows, report)

    # ── 5. Augmentation ───────────────────────────────────────────────────────
    checks["augmentation"] = check_augmentation(rows, report)

    # ── 6. Image integrity ────────────────────────────────────────────────────
    if not skip_images:
        checks["images"] = check_image_integrity(rows, images_dir, report, max_check=args.max_image_check)
    else:
        report.append(f"\n{'─'*60}")
        report.append("6. IMAGE FILE INTEGRITY — SKIPPED (directory not found)")
        checks["images"] = None

    # ── 7. Duplicates ─────────────────────────────────────────────────────────
    checks["duplicates"] = check_duplicates(rows, report)

    # ── 8. Identical src==trg check ───────────────────────────────────────────
    report.append(f"\n{'─'*60}")
    report.append("8. IDENTICAL SRC==TRG (no correction applied)")
    report.append(f"{'─'*60}")
    checks["identical"] = check_identical_pairs(rows, report)

    # ── Summary ───────────────────────────────────────────────────────────────
    report.append(f"\n{'='*60}")
    report.append("SUMMARY")
    report.append(f"{'='*60}")

    pass_count = sum(1 for v in checks.values() if v is True)
    fail_count = sum(1 for v in checks.values() if v is False)
    skip_count = sum(1 for v in checks.values() if v is None)

    for check_name, result in checks.items():
        icon = "✅" if result is True else ("⚠️ " if result is False else "⏭️ ")
        report.append(f"  {icon} {check_name:<20} {'PASS' if result else ('SKIP' if result is None else 'WARN')}")

    report.append(f"\n  Total: {pass_count} passed, {fail_count} warnings, {skip_count} skipped")

    overall = "✅ DATASET LOOKS HEALTHY" if fail_count == 0 else f"⚠️  {fail_count} CHECK(S) NEED ATTENTION"
    report.append(f"\n  {overall}")
    report.append(f"{'='*60}")

    # ── Build sample pairs ────────────────────────────────────────────────────
    sample_lines = build_sample_pairs(samples_per_cat, rows, report, n_per_cat=args.sample_n)

    # ── Print report to console ───────────────────────────────────────────────
    for line in report:
        print(line)

    # ── Save report files ─────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "verification_report.txt")
    sample_path = os.path.join(out_dir, "sample_pairs.txt")

    os.makedirs(out_dir, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n✅  Report saved → {report_path}")

    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_lines))
    print(f"✅  Sample pairs  → {sample_path}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
