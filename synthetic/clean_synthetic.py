"""
clean_synthetic.py — Filter low-quality rows from ITIEC-Syn metadata.

Removes rows where src_text == trg_text (no correction applied).
These rows contribute nothing to model training — the OCR + GEC pipeline
has no signal to learn from when input and target are identical.

Outputs
-------
  metadata_clean.csv     — filtered dataset (original untouched)
  cleaning_report.txt    — before/after statistics per category

Usage
-----
    python clean_synthetic.py --metadata ./output/metadata.csv --out ./output

    # Dry run (show stats, don't write):
    python clean_synthetic.py --metadata ./output/metadata.csv --dry-run
"""

import os
import csv
import sys
import argparse
from collections import defaultdict, Counter
from datetime import datetime


# ── Expected categories (for ordered reporting) ───────────────────────────────
ALL_CATEGORIES = [
    "morphological", "syntactic", "semantic",
    "spelling", "typographic", "algospeak",
]


def _pct(n, total):
    return (n / total * 100) if total > 0 else 0.0


def _bar(n, total, width=25):
    filled = int(round(n / total * width)) if total > 0 else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Filter identical src==trg rows from ITIEC-Syn metadata.csv."
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to metadata.csv (input, not modified)."
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory. Defaults to same directory as metadata.csv."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only, do not write any files."
    )
    parser.add_argument(
        "--output-name", default="metadata_clean.csv",
        help="Filename for cleaned output (default: metadata_clean.csv)."
    )
    args = parser.parse_args()

    meta_path = args.metadata
    out_dir   = args.out or os.path.dirname(os.path.abspath(meta_path))

    if not os.path.exists(meta_path):
        print(f"ERROR: File not found: {meta_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f" ITIEC-Syn Dataset Cleaning")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f" input  : {meta_path}")
    print(f" output : {out_dir}/{args.output_name}")
    print(f" mode   : {'DRY RUN (no files written)' if args.dry_run else 'WRITE'}")
    print(f"{'='*60}\n")

    # ── Stream-read and partition rows ────────────────────────────────────────
    kept_rows   = []
    removed_rows = []
    fieldnames  = None

    # Per-category counters
    cat_total   = Counter()
    cat_removed = Counter()

    print("Reading metadata.csv ...", end=" ", flush=True)
    with open(meta_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            cat = row.get("category", "unknown")
            src = row.get("src_text", "")
            trg = row.get("trg_text", "")

            cat_total[cat] += 1

            if src.strip() == trg.strip():
                cat_removed[cat] += 1
                removed_rows.append(row)
            else:
                kept_rows.append(row)

    total_in    = len(kept_rows) + len(removed_rows)
    total_kept  = len(kept_rows)
    total_removed = len(removed_rows)

    print(f"{total_in:,} rows read.")

    # ── Build report ──────────────────────────────────────────────────────────
    report = []
    report.append("=" * 60)
    report.append(" ITIEC-Syn CLEANING REPORT")
    report.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f" Input    : {meta_path}")
    report.append("=" * 60)

    report.append(f"\n{'─'*60}")
    report.append("BEFORE / AFTER SUMMARY")
    report.append(f"{'─'*60}")
    report.append(f"  Total rows (before) : {total_in:,}")
    report.append(f"  Rows removed        : {total_removed:,}  ({_pct(total_removed, total_in):.1f}%)")
    report.append(f"  Total rows (after)  : {total_kept:,}")

    report.append(f"\n{'─'*60}")
    report.append("PER-CATEGORY BREAKDOWN")
    report.append(f"{'─'*60}")
    report.append(f"  {'Category':<16} {'Before':>8}  {'Removed':>9}  {'Removed%':>9}  {'After':>8}  {'Bar (after)'}")

    for cat in ALL_CATEGORIES:
        n_before  = cat_total.get(cat, 0)
        n_removed = cat_removed.get(cat, 0)
        n_after   = n_before - n_removed
        pct_rem   = _pct(n_removed, n_before)
        flag = "⚠️ " if pct_rem > 5 else "   "
        bar = _bar(n_after, total_kept)
        report.append(
            f"  {flag}{cat:<14} {n_before:>8,}  {n_removed:>9,}  {pct_rem:>8.1f}%  "
            f"{n_after:>8,}  {bar}"
        )

    unknown_cats = set(cat_total.keys()) - set(ALL_CATEGORIES)
    if unknown_cats:
        for cat in unknown_cats:
            n_before  = cat_total.get(cat, 0)
            n_removed = cat_removed.get(cat, 0)
            report.append(f"  ⚠️  {cat} (unknown): before={n_before}, removed={n_removed}")

    report.append(f"\n{'─'*60}")
    report.append("NEW CATEGORY DISTRIBUTION (after cleaning)")
    report.append(f"{'─'*60}")
    report.append(f"  {'Category':<16} {'Count':>8}  {'%':>7}  {'Bar'}")

    for cat in ALL_CATEGORIES:
        n_before  = cat_total.get(cat, 0)
        n_removed = cat_removed.get(cat, 0)
        n_after   = n_before - n_removed
        pct = _pct(n_after, total_kept)
        bar = _bar(n_after, total_kept)
        report.append(f"  {cat:<16} {n_after:>8,}  {pct:>6.1f}%  {bar}")

    report.append(f"\n  Total: {total_kept:,} rows")

    # ── Show removed samples (first 3 per category) ───────────────────────────
    report.append(f"\n{'─'*60}")
    report.append("REMOVED ROW SAMPLES (up to 3 per category)")
    report.append(f"{'─'*60}")

    removed_by_cat = defaultdict(list)
    for row in removed_rows:
        removed_by_cat[row.get("category", "unknown")].append(row)

    for cat in ALL_CATEGORIES:
        samples = removed_by_cat.get(cat, [])[:3]
        if not samples:
            continue
        report.append(f"\n  [{cat.upper()}] — {len(removed_by_cat.get(cat, []))} identical rows total")
        for row in samples:
            src = row.get("src_text", "")
            img = os.path.basename(row.get("image_path", ""))
            # Show only first 80 chars if long
            preview = src[:80] + "..." if len(src) > 80 else src
            report.append(f"    img={img}: \"{preview}\"")

    report.append(f"\n{'='*60}")
    report.append(f" Cleaning complete: {total_removed:,} rows removed → {total_kept:,} kept")
    report.append(f"{'='*60}")

    # ── Print report ──────────────────────────────────────────────────────────
    for line in report:
        print(line)

    if args.dry_run:
        print("\n[DRY RUN] No files written. Remove --dry-run to apply.")
        return

    # ── Write cleaned metadata ────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    out_csv  = os.path.join(out_dir, args.output_name)
    out_rpt  = os.path.join(out_dir, "cleaning_report.txt")

    print(f"\nWriting {total_kept:,} rows to {out_csv} ...", end=" ", flush=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)
    print("done.")

    with open(out_rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"✅  Cleaned metadata → {out_csv}")
    print(f"✅  Cleaning report  → {out_rpt}")
    print(f"\n  Original file untouched: {meta_path}")


if __name__ == "__main__":
    main()
