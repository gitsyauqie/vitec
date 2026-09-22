"""
generate_synthetic.py — Main synthetic data generation script for ITIEC-Syn.

Usage
-----
Basic (5,000 images from IGED):
    python generate_synthetic.py --iged iged.csv --out ./output --n 5000

Full run (100K images, balanced categories):
    python generate_synthetic.py --iged iged.csv --out ./output --n 100000 --balanced

With algospeak injection (adds algospeak variant images):
    python generate_synthetic.py --iged iged.csv --out ./output --n 50000 --algospeak algospeak_dict.json

Resume interrupted run:
    python generate_synthetic.py --iged iged.csv --out ./output --n 50000 --resume

Output format
-------------
./output/
  images/
    img_000001.png
    img_000002.png
    ...
  metadata.csv   <- columns: image_id, image_path, src_text, trg_text, category, augmentation, split
  run_config.json
"""

import os
import json
import random
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Local modules (must be in ./utils/)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
from renderer import render_text_image, FONT_PATHS
from augmentor import augment


# ── Category configuration ────────────────────────────────────────────────────
# IGED actual categories (verified from dataset):
#   afiksasi (19.6%), sintaksis_frasa (18.8%), preposisi (12.5%),
#   pembentukan_kata (12.2%), kelengkapan_kalimat (11.8%), diksi (10.5%),
#   reduplikasi (6.7%), ambigu (4.9%), pleonasme (2.0%),
#   pleonasme_claude (0.3%), diksi_claude (0.3%), ambigu_claude (0.3%)
#
# NOTE: IGED has NO spelling/typographic categories.
#       Those are generated synthetically via inject_spelling_errors().

CATEGORY_MAP = {
    # Morphological (afiksasi=affixation, pembentukan_kata=word formation,
    #                reduplikasi=reduplication)
    "afiksasi":             "morphological",
    "pembentukan_kata":     "morphological",
    "reduplikasi":          "morphological",
    # Syntactic (sintaksis_frasa=phrase syntax, preposisi=preposition,
    #            kelengkapan_kalimat=sentence completeness)
    "sintaksis_frasa":      "syntactic",
    "preposisi":            "syntactic",
    "kelengkapan_kalimat":  "syntactic",
    # Semantic (diksi=diction, ambigu=ambiguity, pleonasme=pleonasm)
    "diksi":                "semantic",
    "ambigu":               "semantic",
    "pleonasme":            "semantic",
    "pleonasme_claude":     "semantic",
    "diksi_claude":         "semantic",
    "ambigu_claude":        "semantic",
}

# IGED actual distribution → Paper 5 category grouping:
#   morphological : afiksasi + pembentukan_kata + reduplikasi  ≈ 38.5%
#   syntactic     : sintaksis_frasa + preposisi + kelengkapan  ≈ 43.2%
#   semantic      : diksi + ambigu + pleonasme + *_claude      ≈ 18.3%
#   spelling      : NOT in IGED — generated via inject_spelling_errors()
#   typographic   : NOT in IGED — generated via inject_spelling_errors()
#   algospeak     : NOT in IGED — generated via inject_algospeak()

# Sampling budget: IGED-derived categories use proportional weights;
# spelling, typographic, algospeak are fully synthetic.
DEFAULT_CATEGORY_WEIGHTS = {
    "morphological":  0.25,   # from IGED (afiksasi+pembentukan_kata+reduplikasi)
    "syntactic":      0.25,   # from IGED (sintaksis_frasa+preposisi+kelengkapan_kalimat)
    "semantic":       0.15,   # from IGED (diksi+ambigu+pleonasme)
    "spelling":       0.15,   # synthetic — inject_spelling_errors() on trg text
    "typographic":    0.10,   # synthetic — inject_spelling_errors(mode='typo') on trg text
    "algospeak":      0.10,   # synthetic — inject_algospeak() on trg text
}

# Augmentation intensity per category
AUGMENTATION_INTENSITY = {
    "morphological":  "medium",
    "syntactic":      "medium",
    "semantic":       "medium",
    "spelling":       "medium",
    "typographic":    "light",   # keep typos legible for the model
    "algospeak":      "heavy",   # social-media photos are noisier
}


# ── Spelling & typographic error injection ───────────────────────────────────
# These simulate errors NOT present in IGED but common in real images.
#
# Two modes:
#   'spelling'    — realistic misspellings (transposition, substitution, omission)
#   'typographic' — keyboard/OCR-style typos (adjacent keys, double chars)

# Indonesian keyboard adjacency map (QWERTY)
_ADJACENT_KEYS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'aqwdxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx',
}

# Common Indonesian misspelling patterns
_INDONESIAN_CONFUSABLES = [
    # vowel confusion
    ('ai', 'ay'), ('au', 'aw'), ('ei', 'ey'),
    # consonant doubling/single
    ('kk', 'k'), ('ll', 'l'), ('nn', 'n'), ('ss', 's'), ('tt', 't'),
    ('k', 'kk'), ('l', 'll'), ('n', 'nn'), ('s', 'ss'), ('t', 'tt'),
    # common Indonesian swaps
    ('f', 'v'), ('v', 'f'), ('z', 's'), ('s', 'z'),
    ('ny', 'ni'), ('ng', 'n'), ('kh', 'k'),
    ('ph', 'f'), ('th', 't'),
    # prefix/suffix common errors
    ('mem', 'mem'), ('men', 'meng'), ('meng', 'men'),
    ('ke', 'ke-'), ('di', 'di-'),
    ('kan', 'in'), ('an', 'nya'),
]


def inject_spelling_errors(
    text: str,
    mode: str = 'spelling',
    n_errors: int | None = None,
    rng=None,
) -> tuple[str, str]:
    """
    Inject spelling or typographic errors into *text*.

    Parameters
    ----------
    text : str
        Clean source text (typically from IGED 'trg' column).
    mode : str
        'spelling'    — realistic misspellings (omission, transposition, substitution)
        'typographic' — keyboard adjacency typos, double-chars, OCR confusables
    n_errors : int, optional
        Number of words to corrupt. Defaults to 1–3 based on text length.
    rng : random.Random, optional

    Returns
    -------
    (src_with_errors, trg_clean) : tuple[str, str]
    """
    rng = rng or random
    words = text.split()
    if len(words) < 2:
        return text, text

    if n_errors is None:
        n_errors = rng.randint(1, min(3, max(1, len(words) // 5)))

    modified = words[:]
    # Pick random word indices (skip very short words < 4 chars)
    candidates = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidates:
        return text, text
    rng.shuffle(candidates)
    targets = candidates[:n_errors]

    for idx in targets:
        word = words[idx]
        corrupted = _corrupt_word(word, mode=mode, rng=rng)
        modified[idx] = corrupted

    return " ".join(modified), text


def _corrupt_word(word: str, mode: str, rng) -> str:
    """Apply a single corruption to *word*."""
    # Preserve leading/trailing punctuation
    prefix, core, suffix = "", word, ""
    while core and not core[0].isalpha():
        prefix += core[0]; core = core[1:]
    while core and not core[-1].isalpha():
        suffix = core[-1] + suffix; core = core[:-1]
    if len(core) < 3:
        return word

    if mode == 'typographic':
        op = rng.choice(['adjacent_key', 'double_char', 'swap_adjacent', 'ocr_confusable'])
    else:  # spelling
        op = rng.choice(['omit_char', 'transpose', 'substitute_vowel', 'indonesian_pattern'])

    result = core

    if op == 'adjacent_key':
        pos = rng.randint(0, len(core) - 1)
        ch = core[pos].lower()
        if ch in _ADJACENT_KEYS:
            replacement = rng.choice(_ADJACENT_KEYS[ch])
            # Preserve original case
            if core[pos].isupper():
                replacement = replacement.upper()
            result = core[:pos] + replacement + core[pos+1:]

    elif op == 'double_char':
        pos = rng.randint(0, len(core) - 1)
        result = core[:pos] + core[pos] + core[pos:]   # duplicate one char

    elif op == 'swap_adjacent':
        if len(core) >= 2:
            pos = rng.randint(0, len(core) - 2)
            lst = list(core)
            lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
            result = ''.join(lst)

    elif op == 'ocr_confusable':
        # Common OCR errors for Latin script
        ocr_pairs = [('l','1'), ('0','o'), ('1','i'), ('rn','m'),
                     ('cl','d'), ('vv','w'), ('li','h')]
        rng.shuffle(ocr_pairs)
        for src_ch, tgt_ch in ocr_pairs:
            if src_ch in core.lower():
                result = core.lower().replace(src_ch, tgt_ch, 1)
                # Restore original capitalisation
                if core[0].isupper():
                    result = result.capitalize()
                break

    elif op == 'omit_char':
        pos = rng.randint(1, len(core) - 2)   # never omit first/last
        result = core[:pos] + core[pos+1:]

    elif op == 'transpose':
        if len(core) >= 2:
            pos = rng.randint(0, len(core) - 2)
            lst = list(core)
            lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
            result = ''.join(lst)

    elif op == 'substitute_vowel':
        vowels = 'aeiou'
        positions = [i for i, c in enumerate(core.lower()) if c in vowels]
        if positions:
            pos = rng.choice(positions)
            current = core[pos].lower()
            other_vowels = [v for v in vowels if v != current]
            replacement = rng.choice(other_vowels)
            if core[pos].isupper():
                replacement = replacement.upper()
            result = core[:pos] + replacement + core[pos+1:]

    elif op == 'indonesian_pattern':
        rng.shuffle(_INDONESIAN_CONFUSABLES)
        for src_pat, tgt_pat in _INDONESIAN_CONFUSABLES:
            if src_pat in core.lower():
                result = core.lower().replace(src_pat, tgt_pat, 1)
                if core[0].isupper():
                    result = result[0].upper() + result[1:]
                break

    return prefix + result + suffix


# ── Algospeak injection ───────────────────────────────────────────────────────

def load_algospeak_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Flatten all substitution rules into a single dict: formal → [informal, ...]
    flat = {}
    for key in ("abbreviations", "phonetic_substitutions"):
        if key in data:
            for formal, variants in data[key].items():
                if isinstance(variants, list):
                    flat[formal] = variants
    if "combined_rules" in data:
        for rule in data["combined_rules"]:
            if isinstance(rule, dict) and "from" in rule and "to" in rule:
                flat[rule["from"]] = rule["to"] if isinstance(rule["to"], list) else [rule["to"]]
    return flat


def inject_algospeak(text: str, algo_dict: dict, n_substitutions: int = 2, rng=None) -> tuple[str, str]:
    """
    Replace n_substitutions words in *text* with algospeak variants.
    Returns (src_algospeak, trg_clean) pair.
    """
    rng = rng or random
    words = text.split()
    modified = words[:]
    substituted = 0
    indices = list(range(len(words)))
    rng.shuffle(indices)
    for idx in indices:
        word = words[idx].lower().rstrip(".,!?;:")
        if word in algo_dict and algo_dict[word]:
            replacement = rng.choice(algo_dict[word])
            # Preserve trailing punctuation
            punct = ""
            for p in ".,!?;:":
                if words[idx].endswith(p):
                    punct = p
                    break
            modified[idx] = replacement + punct
            substituted += 1
            if substituted >= n_substitutions:
                break
    src = " ".join(modified)
    trg = text  # original clean text is the target
    return src, trg


# ── Memory-efficient IGED loader (reservoir sampling) ────────────────────────

def _reservoir_sample_iged(
    csv_path: str,
    category_map: dict,
    n_per_category: dict,
    n_clean: int,
    chunk_size: int = 50_000,
    seed: int = 42,
) -> tuple[dict, dict, list]:
    """
    Read IGED CSV in chunks of `chunk_size` rows — never loads the full file.

    Uses reservoir sampling (Vitter's Algorithm R) to maintain a fixed-size
    sample per category without keeping all rows in memory.

    Peak memory ≈ chunk_size × row_bytes + sum(n_per_category) × row_bytes
                ≈ much less than loading the full 1.3M CSV at once.

    Parameters
    ----------
    csv_path        : path to IGED CSV
    category_map    : IGED raw category → normalised label
    n_per_category  : {'morphological': N, 'syntactic': N, 'semantic': N}
    n_clean         : how many clean (trg) sentences to collect for synthetic phases
    chunk_size      : rows per chunk (default 50K — tune down if still OOM)
    seed            : random seed

    Returns
    -------
    reservoirs   : {category: [(src, trg), ...]}   — sampled IGED pairs
    cat_counts   : {category: int}                 — total seen per category
    clean_pool   : [str]                           — clean trg sentences
    """
    import collections
    rng = random.Random(seed)

    all_cats = list(n_per_category.keys())
    reservoirs  = {cat: [] for cat in all_cats}
    cat_counts  = collections.defaultdict(int)
    clean_pool  = []
    clean_count = 0

    chunk_iter = pd.read_csv(
        csv_path,
        usecols=["src", "trg", "category", "split"],
        chunksize=chunk_size,
        dtype=str,
        on_bad_lines="skip",
        engine="c",
    )

    total_seen = 0
    for chunk_num, chunk in enumerate(chunk_iter):
        # Normalise column names
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Filter: train split only
        if "split" in chunk.columns:
            chunk = chunk[chunk["split"].str.strip() == "train"]

        # Map raw category → normalised
        chunk = chunk.copy()   # avoid SettingWithCopyWarning
        chunk["cat_norm"] = (
            chunk["category"].str.strip().str.lower().map(category_map)
        )

        # Drop malformed rows
        chunk = chunk.dropna(subset=["cat_norm", "src", "trg"])
        chunk = chunk[chunk["src"].str.strip() != ""]
        chunk = chunk[chunk["trg"].str.strip() != ""]

        total_seen += len(chunk)
        print(f"  chunk {chunk_num+1:>4}: {len(chunk):>6,} valid rows  "
              f"(total seen: {total_seen:,})", end="\r")

        # ── Reservoir sampling per category ──────────────────────────────
        for cat in all_cats:
            cat_rows = chunk[chunk["cat_norm"] == cat]
            n_target = n_per_category[cat]
            for _, row in cat_rows.iterrows():
                cat_counts[cat] += 1
                k = cat_counts[cat]
                pair = (str(row["src"]).strip(), str(row["trg"]).strip())
                if len(reservoirs[cat]) < n_target:
                    reservoirs[cat].append(pair)
                else:
                    # Replace with probability n_target / k
                    j = rng.randint(0, k - 1)
                    if j < n_target:
                        reservoirs[cat][j] = pair

        # ── Collect clean sentences for synthetic phases ──────────────────
        if clean_count < n_clean:
            for trg in chunk["trg"].str.strip().tolist():
                if trg and clean_count < n_clean:
                    clean_pool.append(trg)
                    clean_count += 1

    print()  # newline after \r
    print(f"  Total IGED training rows scanned: {total_seen:,}")
    return reservoirs, dict(cat_counts), clean_pool


# ── Main generation logic ─────────────────────────────────────────────────────

def generate_dataset(args):
    # ── Setup output directories ──────────────────────────────────────────
    out_root = Path(args.out)
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_root / "metadata.csv"
    config_path = out_root / "run_config.json"

    # ── Save run config ───────────────────────────────────────────────────
    run_config = vars(args)
    run_config["timestamp"] = datetime.now().isoformat()
    run_config["font_count"] = len(FONT_PATHS)
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # ── Determine per-category sample counts (needed before loading) ──────
    n_total = args.n
    weights = DEFAULT_CATEGORY_WEIGHTS
    category_budgets_pre = {k: int(n_total * weights[k]) for k in weights}
    diff_pre = n_total - sum(category_budgets_pre.values())
    if diff_pre != 0:
        category_budgets_pre["morphological"] += diff_pre

    iged_budgets = {
        k: category_budgets_pre[k]
        for k in ["morphological", "syntactic", "semantic"]
    }

    # ── Load IGED via reservoir sampling (memory-efficient for 1M+ rows) ──
    print(f"Loading IGED from: {args.iged}")
    print(f"  Strategy: reservoir sampling ({args.chunk_size:,} rows/chunk) — avoids OOM")
    reservoirs, cat_counts, clean_reservoir = _reservoir_sample_iged(
        csv_path=args.iged,
        category_map=CATEGORY_MAP,
        n_per_category=iged_budgets,
        n_clean=category_budgets_pre.get("spelling", 0)
               + category_budgets_pre.get("typographic", 0)
               + category_budgets_pre.get("algospeak", 0)
               + 5000,   # buffer for clean sentences
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    print("\n  Rows sampled per IGED category:")
    for cat, rows in reservoirs.items():
        print(f"    {cat:<20} {len(rows):>6,}  (seen in IGED: {cat_counts.get(cat, 0):,})")
    print(f"  Clean sentences for synthetic: {len(clean_reservoir):,}")

    # ── Load algospeak dict ───────────────────────────────────────────────
    algo_dict = {}
    if args.algospeak and os.path.exists(args.algospeak):
        algo_dict = load_algospeak_dict(args.algospeak)
        print(f"  Algospeak rules loaded: {len(algo_dict):,}")

    # category_budgets already computed above as category_budgets_pre
    category_budgets = category_budgets_pre
    IGED_CATS = ["morphological", "syntactic", "semantic"]

    print("\n  Sampling budget:")
    for cat, n in category_budgets.items():
        source = "(IGED)" if cat in IGED_CATS else "(synthetic)"
        print(f"    {cat:<20} {n:>6,}  {source}")
    print(f"    {'TOTAL':<20} {sum(category_budgets.values()):>6,}")

    # ── Determine resume offset ───────────────────────────────────────────
    start_idx = 0
    existing_rows = 0
    if args.resume and metadata_path.exists():
        # Only read 1 column to count rows — very low memory
        existing_rows = sum(1 for _ in open(metadata_path)) - 1  # subtract header
        start_idx = existing_rows
        print(f"\nResuming from image #{start_idx:,}")

    # ── Open metadata CSV ─────────────────────────────────────────────────
    csv_mode = "a" if args.resume and metadata_path.exists() else "w"
    csv_file = open(metadata_path, csv_mode, newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "image_id", "image_path", "src_text", "trg_text",
        "category", "augmentation_intensity", "font", "split"
    ])
    if csv_mode == "w":
        csv_writer.writeheader()

    # ── Generate images ───────────────────────────────────────────────────
    rng = random.Random(args.seed)
    global_idx = start_idx
    total_generated = existing_rows

    def _save_image(src, trg, category, img_idx, font_path):
        aug_intensity = AUGMENTATION_INTENSITY.get(category, "medium")
        img = render_text_image(src, font_path=font_path, seed=img_idx)
        img = augment(img, seed=img_idx, intensity=aug_intensity)
        img_filename = f"img_{img_idx:07d}.png"
        img.save(str(img_dir / img_filename), format="PNG", optimize=True)
        csv_writer.writerow({
            "image_id":               img_idx,
            "image_path":             f"images/{img_filename}",
            "src_text":               src,
            "trg_text":               trg,
            "category":               category,
            "augmentation_intensity": aug_intensity,
            "font":                   os.path.basename(font_path) if font_path else "default",
            "split":                  "train",
        })

    # ── 1. IGED-derived categories (morphological / syntactic / semantic) ─
    print("\n── Phase 1: IGED-derived images ──")
    for category in ["morphological", "syntactic", "semantic"]:
        pairs = reservoirs.get(category, [])
        if not pairs:
            print(f"  WARNING: no rows sampled for '{category}' — skipping")
            continue
        # If reservoir is smaller than budget, sample with replacement
        budget = category_budgets.get(category, 0)
        if len(pairs) < budget:
            pairs = rng.choices(pairs, k=budget)
        pbar = tqdm(pairs, total=len(pairs), desc=f"  [{category}]", leave=True)
        for src, trg in pbar:
            font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
            _save_image(src, trg, category, global_idx, font_path)
            global_idx += 1; total_generated += 1
            if total_generated % 500 == 0:
                csv_file.flush()

    # ── 2. Spelling & typographic (inject errors into IGED clean text) ────
    print("\n── Phase 2: Synthetic spelling/typographic images ──")
    # Use clean_reservoir collected during IGED scan (no df needed)
    clean_sentences = clean_reservoir

    for mode, category in [("spelling", "spelling"), ("typographic", "typographic")]:
        budget = category_budgets.get(category, 0)
        if budget <= 0:
            continue
        sample_sents = rng.choices(clean_sentences, k=budget)
        pbar = tqdm(sample_sents, desc=f"  [{category}]", leave=True)
        for clean_text in pbar:
            n_err = rng.randint(1, 3)
            src, trg = inject_spelling_errors(clean_text, mode=mode, n_errors=n_err, rng=rng)
            font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
            _save_image(src, trg, category, global_idx, font_path)
            global_idx += 1; total_generated += 1
            if total_generated % 500 == 0:
                csv_file.flush()

    # ── 3. Algospeak images ───────────────────────────────────────────────
    algo_budget = category_budgets.get("algospeak", 0)
    if algo_budget > 0 and algo_dict:
        print(f"\n── Phase 3: Algospeak images ({algo_budget:,}) ──")
        base_sentences = rng.choices(clean_sentences, k=algo_budget)

        pbar = tqdm(base_sentences, desc="  [algospeak]", leave=True)
        for clean_text in pbar:
            n_subs = rng.randint(1, 3)
            src, trg = inject_algospeak(clean_text, algo_dict, n_substitutions=n_subs, rng=rng)

            font_path = rng.choice(FONT_PATHS) if FONT_PATHS else None
            img = render_text_image(src, font_path=font_path, seed=global_idx)
            img = augment(img, seed=global_idx, intensity="heavy")

            img_filename = f"img_{global_idx:07d}.png"
            img_path = img_dir / img_filename
            img.save(str(img_path), format="PNG", optimize=True)

            csv_writer.writerow({
                "image_id":             global_idx,
                "image_path":           f"images/{img_filename}",
                "src_text":             src,
                "trg_text":             trg,
                "category":             "algospeak",
                "augmentation_intensity": "heavy",
                "font":                 os.path.basename(font_path) if font_path else "default",
                "split":                "train",
            })

            global_idx += 1
            total_generated += 1

            if total_generated % 500 == 0:
                csv_file.flush()

    csv_file.close()

    # ── Final summary — count lines instead of loading full CSV ───────────
    total_imgs = sum(1 for _ in open(metadata_path)) - 1
    # Read only category column for distribution (low memory)
    cat_counts_final = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cat_col = header.index("category")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > cat_col:
                cat = parts[cat_col].strip('"')
                cat_counts_final[cat] = cat_counts_final.get(cat, 0) + 1

    print("\n" + "=" * 50)
    print(f"✓ Generation complete!")
    print(f"  Total images:   {total_imgs:,}")
    print(f"  Output folder:  {out_root}")
    print(f"  Metadata:       {metadata_path}")
    print("\nCategory distribution:")
    for cat, count in sorted(cat_counts_final.items(), key=lambda x: -x[1]):
        pct = count / total_imgs * 100 if total_imgs else 0
        print(f"  {cat:<22} {count:>6,}  ({pct:.1f}%)")
    print("=" * 50)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="ITIEC-Syn: Synthetic data generator for Indonesian Text-in-Image Error Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--iged",        required=True,   help="Path to IGED .csv file")
    p.add_argument("--out",         required=True,   help="Output directory")
    p.add_argument("--n",           type=int, default=10000,
                   help="Total images to generate")
    p.add_argument("--algospeak",   default=None,    help="Path to algospeak_dict.json")
    p.add_argument("--chunk-size",  type=int, default=50_000, dest="chunk_size",
                   help="CSV rows per chunk during IGED loading. "
                        "Reduce to 10000-20000 if OOM on low-RAM servers.")
    p.add_argument("--resume",      action="store_true",
                   help="Resume interrupted generation")
    p.add_argument("--seed",        type=int, default=42, help="Random seed")
    # kept for backward compat
    p.add_argument("--balanced",    action="store_true",
                   help="(Ignored — category weights are fixed in DEFAULT_CATEGORY_WEIGHTS)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    generate_dataset(args)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
