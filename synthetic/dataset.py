"""
dataset.py — ITIEC GEC Training Dataset Pipeline

Mendukung dua mode utama:
  1. Fairseq mode  : generate train/valid/test .src & .tgt plain-text files
                     → langsung di-feed ke `fairseq-preprocess` untuk fine-tune CASTLE
  2. PyTorch mode  : ITIECSynDataset / ITIECRealDataset sebagai torch.utils.data.Dataset
                     → untuk custom training loop atau pipeline multimodal (image + text)

Alur pipeline Paper 5 (ITIEC):
  Image ──► PP-OCRv4 (M1) ──► src_text / OCR-hyp ──► CASTLE (M3) ──► corrected_text
                                                           ▲
                                               dataset.py menyiapkan data ini

Penggunaan cepat
----------------
  # 1. Buat file Fairseq untuk fine-tune CASTLE:
  python dataset.py prepare \\
      --metadata /ssd-data1/sq2023/Paper5/ITIEC_Synthetic/output/metadata_clean.csv \\
      --out      /ssd-data1/sq2023/Paper5/ITIEC_GEC/data-bin/raw \\
      --val-ratio 0.05 --test-ratio 0.05

  # 2. Gunakan sebagai PyTorch Dataset:
  from dataset import ITIECSynDataset, create_dataloaders
  train_loader, val_loader = create_dataloaders(
      metadata_path='.../metadata_clean.csv',
      images_dir='.../images',
      batch_size=128,
  )

Kompatibilitas CASTLE
---------------------
  - Tokenizer   : whitespace (sama persis dengan CASTLE/IGED training)
  - Kategori    : 6 (ITIEC) → mapping ke 3 (CASTLE KG gate: morph/synt/sem)
  - KG gate cat : diksi / ambigu / pleonasme  (untuk semantic sub-type)
  - Batch size  : 128 tokens per sample, max_tokens=1024 per batch (sesuai Table 8)
  - FP16        : didukung via AMP, aktifkan di train.py
"""

import os
import csv
import sys
import json
import random
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

# 6 kategori ITIEC-Syn
ITIEC_CATEGORIES = [
    "morphological", "syntactic", "semantic",
    "spelling", "typographic", "algospeak",
]
ITIEC_CAT2ID = {c: i for i, c in enumerate(ITIEC_CATEGORIES)}

# Mapping 6 ITIEC → 3 CASTLE (untuk KG gating layer)
# Dasar: spelling/typographic = karakter-level ≈ morphological
#        algospeak = substitusi makna ≈ semantic
ITIEC_TO_CASTLE = {
    "morphological": "morphological",
    "syntactic":     "syntactic",
    "semantic":      "semantic",
    "spelling":      "morphological",
    "typographic":   "morphological",
    "algospeak":     "semantic",
}

# Mapping CASTLE-3 → IGED semantic sub-type (untuk p_cat di KG confidence gate)
# Hanya semantic errors yang punya sub-type KG gate
CASTLE_KG_SUBTYPE = {
    "morphological": None,
    "syntactic":     None,
    "semantic":      "diksi",   # default; override jika sub-type diketahui
}

CASTLE_CATEGORIES = ["morphological", "syntactic", "semantic"]
CASTLE_CAT2ID     = {c: i for i, c in enumerate(CASTLE_CATEGORIES)}

# Token khusus (kompatibel dengan BART/Fairseq)
PAD_TOKEN   = "<pad>"
BOS_TOKEN   = "<s>"
EOS_TOKEN   = "</s>"
UNK_TOKEN   = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Default image transform untuk multimodal mode
_DEFAULT_IMG_SIZE = 224


# ─────────────────────────────────────────────────────────────────────────────
#  WhitespaceTokenizer
# ─────────────────────────────────────────────────────────────────────────────

class WhitespaceTokenizer:
    """
    Tokenizer whitespace murni — identik dengan yang dipakai CASTLE/IGED.

    Vocab dibangun secara dinamis dari corpus training.
    Jika CASTLE checkpoint sudah ada, gunakan vocab dari situ (load_vocab).

    Contoh:
        tok = WhitespaceTokenizer()
        tok.build_vocab([("saya makan nasi", "saya makan nasi goreng")])
        ids = tok.encode("saya makan")   # → [5, 6]
        txt = tok.decode(ids)            # → "saya makan"
    """

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._built = False
        self._init_specials()

    def _init_specials(self):
        for tok in SPECIAL_TOKENS:
            idx = len(self.token2id)
            self.token2id[tok] = idx
            self.id2token[idx] = tok

    def tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return text.strip().split()

    def build_vocab(self, pairs: List[Tuple[str, str]], min_freq: int = 1):
        """Bangun vocabulary dari list pasangan (src, trg)."""
        freq: Counter = Counter()
        for src, trg in pairs:
            freq.update(self.tokenize(src))
            freq.update(self.tokenize(trg))

        for token, count in sorted(freq.items(), key=lambda x: -x[1]):
            if count >= min_freq and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token

        self._built = True
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        tokens = self.tokenize(text)
        if max_length:
            tokens = tokens[: max_length - int(add_bos) - int(add_eos)]
        ids = [self.token2id.get(t, self.token2id[UNK_TOKEN]) for t in tokens]
        if add_bos:
            ids = [self.token2id[BOS_TOKEN]] + ids
        if add_eos:
            ids = ids + [self.token2id[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = []
        for i in ids:
            t = self.id2token.get(i, UNK_TOKEN)
            if skip_special and t in SPECIAL_TOKENS:
                continue
            tokens.append(t)
        return " ".join(tokens)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"token2id": self.token2id, "lowercase": self.lowercase}, f,
                      ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WhitespaceTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(lowercase=data.get("lowercase", False))
        tok.token2id = data["token2id"]
        tok.id2token = {int(v): k for k, v in data["token2id"].items()}
        tok._built = True
        return tok

    def __len__(self):
        return len(self.token2id)

    @property
    def pad_id(self):  return self.token2id[PAD_TOKEN]
    @property
    def bos_id(self):  return self.token2id[BOS_TOKEN]
    @property
    def eos_id(self):  return self.token2id[EOS_TOKEN]
    @property
    def unk_id(self):  return self.token2id[UNK_TOKEN]


# ─────────────────────────────────────────────────────────────────────────────
#  ITIECSynDataset  (Synthetic — metadata_clean.csv)
# ─────────────────────────────────────────────────────────────────────────────

class ITIECSynDataset(Dataset):
    """
    Dataset untuk ITIEC-Syn (metadata_clean.csv).

    Parameter
    ---------
    metadata_path : str
        Path ke metadata_clean.csv
    images_dir : str, optional
        Root direktori gambar. Diperlukan jika mode='multimodal'.
        Jika None, image_path dari CSV dipakai as-is (absolute path).
    tokenizer : WhitespaceTokenizer, optional
        Jika None, teks dikembalikan sebagai string (berguna untuk Fairseq prep).
    split : 'train' | 'val' | 'test' | 'all'
        Split yang diinginkan. Jika CSV tidak punya kolom 'split',
        data di-split otomatis menggunakan stratified sampling.
    val_ratio : float
        Proporsi data untuk validasi (default 0.05).
    test_ratio : float
        Proporsi data untuk test (default 0.05).
    seed : int
        Random seed untuk reproduksibilitas.
    categories : list, optional
        Filter hanya kategori tertentu. None = semua kategori.
    mode : 'text_only' | 'multimodal'
        'text_only'  → kembalikan (src_ids, trg_ids, castle_cat_id, itiec_cat_id)
        'multimodal' → tambahkan pixel_values dari gambar
    use_ocr_hyp : bool
        Jika True, gunakan kolom 'ocr_hyp' sebagai input (bukan src_text).
        Kolom ini ada jika run_ocr_eval.py sudah digabung ke metadata.
    max_src_len : int
        Panjang maksimum token input (default 128).
    max_trg_len : int
        Panjang maksimum token target (default 128).
    """

    def __init__(
        self,
        metadata_path: str,
        images_dir: Optional[str] = None,
        tokenizer: Optional[WhitespaceTokenizer] = None,
        split: str = "train",
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 42,
        categories: Optional[List[str]] = None,
        mode: str = "text_only",
        use_ocr_hyp: bool = False,
        max_src_len: int = 128,
        max_trg_len: int = 128,
    ):
        self.metadata_path = metadata_path
        self.images_dir    = images_dir
        self.tokenizer     = tokenizer
        self.split         = split
        self.mode          = mode
        self.use_ocr_hyp   = use_ocr_hyp
        self.max_src_len   = max_src_len
        self.max_trg_len   = max_trg_len
        self.categories    = categories or ITIEC_CATEGORIES

        # ── Load CSV ──────────────────────────────────────────────────────────
        all_rows = _load_metadata(metadata_path)

        # ── Category filter ───────────────────────────────────────────────────
        all_rows = [r for r in all_rows if r["category"] in self.categories]

        # ── Split ─────────────────────────────────────────────────────────────
        if split != "all":
            all_rows = _stratified_split(
                all_rows, split, val_ratio, test_ratio, seed
            )

        self.data = all_rows

        # ── Image transform (multimodal mode) ─────────────────────────────────
        if mode == "multimodal":
            if not PIL_AVAILABLE:
                raise ImportError("Pillow diperlukan untuk mode='multimodal': pip install Pillow")
            if not TORCHVISION_AVAILABLE:
                raise ImportError("torchvision diperlukan untuk mode='multimodal': pip install torchvision")
            self.img_transform = _default_img_transform()
        else:
            self.img_transform = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]

        # ── Teks input (src) ─────────────────────────────────────────────────
        if self.use_ocr_hyp and "ocr_hyp" in row and row["ocr_hyp"]:
            src_text = row["ocr_hyp"]
        else:
            src_text = row["src_text"]

        trg_text = row["trg_text"]
        category = row["category"]

        # ── Category IDs ──────────────────────────────────────────────────────
        itiec_cat_id  = ITIEC_CAT2ID.get(category, 0)
        castle_cat    = ITIEC_TO_CASTLE.get(category, "morphological")
        castle_cat_id = CASTLE_CAT2ID[castle_cat]

        # ── Tokenize ──────────────────────────────────────────────────────────
        if self.tokenizer is not None:
            src_ids = self.tokenizer.encode(
                src_text, add_bos=False, add_eos=True, max_length=self.max_src_len
            )
            trg_ids = self.tokenizer.encode(
                trg_text, add_bos=True, add_eos=True, max_length=self.max_trg_len
            )
            sample = {
                "src_ids":      torch.tensor(src_ids,     dtype=torch.long),
                "trg_ids":      torch.tensor(trg_ids,     dtype=torch.long),
                "castle_cat":   torch.tensor(castle_cat_id, dtype=torch.long),
                "itiec_cat":    torch.tensor(itiec_cat_id,  dtype=torch.long),
                "src_len":      len(src_ids),
                "trg_len":      len(trg_ids),
            }
        else:
            # Tokenizer None → kembalikan string (untuk Fairseq / debugging)
            sample = {
                "src_text":   src_text,
                "trg_text":   trg_text,
                "castle_cat": castle_cat,
                "itiec_cat":  category,
                "image_path": row.get("image_path", ""),
            }

        # ── Gambar (multimodal mode) ───────────────────────────────────────────
        if self.mode == "multimodal":
            img_path = self._resolve_image_path(row.get("image_path", ""))
            try:
                img = Image.open(img_path).convert("RGB")
                sample["pixel_values"] = self.img_transform(img)
            except Exception as e:
                # Fallback: black image jika file tidak ada
                sample["pixel_values"] = torch.zeros(3, _DEFAULT_IMG_SIZE, _DEFAULT_IMG_SIZE)

        return sample

    def _resolve_image_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path):
            return rel_path
        if self.images_dir:
            return os.path.join(self.images_dir, os.path.basename(rel_path))
        return rel_path

    def get_text_pairs(self) -> List[Tuple[str, str]]:
        """Kembalikan semua pasangan (src_text, trg_text) sebagai list — untuk build vocab."""
        return [(r["src_text"], r["trg_text"]) for r in self.data]

    def category_distribution(self) -> Dict[str, int]:
        """Hitung distribusi per kategori."""
        return dict(Counter(r["category"] for r in self.data))


# ─────────────────────────────────────────────────────────────────────────────
#  ITIECRealDataset  (Real annotated images — untuk T-002 dst.)
# ─────────────────────────────────────────────────────────────────────────────

class ITIECRealDataset(ITIECSynDataset):
    """
    Dataset untuk ITIEC-Real (gambar anotasi manual).

    Perbedaan dari ITIECSynDataset:
    - Reference untuk evaluasi: trg_text (corrected ground truth)
    - Semua split digunakan dari CSV langsung (kolom 'split': train/val/test)
    - Jika kolom 'split' tidak ada, default ke auto-split

    Semua parameter sama dengan ITIECSynDataset.
    Gunakan ini setelah 2000 real images selesai dianotasi.
    """
    pass   # Inherits semua fungsi ITIECSynDataset


# ─────────────────────────────────────────────────────────────────────────────
#  Collator (collate_fn untuk DataLoader)
# ─────────────────────────────────────────────────────────────────────────────

class ITIECCollator:
    """
    Pad dan batch tensor dari ITIECSynDataset.

    Dipakai sebagai collate_fn di DataLoader:
        loader = DataLoader(ds, batch_size=128, collate_fn=ITIECCollator(pad_id))

    Output batch keys:
        src_ids       : (B, T_src)  — padded
        trg_ids       : (B, T_trg)  — padded
        src_mask      : (B, T_src)  — 1 = real token, 0 = pad
        trg_mask      : (B, T_trg)
        castle_cat    : (B,)        — 0=morph, 1=synt, 2=sem
        itiec_cat     : (B,)        — 0..5
        pixel_values  : (B, 3, H, W) — hanya jika multimodal mode
    """

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Cek apakah sudah ditokenisasi (ada 'src_ids') atau masih string
        if "src_ids" not in batch[0]:
            raise ValueError(
                "Collator memerlukan tokenizer. "
                "Set tokenizer=<WhitespaceTokenizer> pada Dataset, "
                "atau gunakan prepare_fairseq_files() untuk Fairseq mode."
            )

        src_list = [b["src_ids"] for b in batch]
        trg_list = [b["trg_ids"] for b in batch]

        src_padded, src_mask = _pad_sequence(src_list, self.pad_id)
        trg_padded, trg_mask = _pad_sequence(trg_list, self.pad_id)

        result = {
            "src_ids":    src_padded,
            "trg_ids":    trg_padded,
            "src_mask":   src_mask,
            "trg_mask":   trg_mask,
            "castle_cat": torch.stack([b["castle_cat"] for b in batch]),
            "itiec_cat":  torch.stack([b["itiec_cat"]  for b in batch]),
        }

        if "pixel_values" in batch[0]:
            result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])

        return result


# ─────────────────────────────────────────────────────────────────────────────
#  create_dataloaders — Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders(
    metadata_path: str,
    images_dir: Optional[str] = None,
    tokenizer: Optional[WhitespaceTokenizer] = None,
    batch_size: int = 128,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    num_workers: int = 4,
    mode: str = "text_only",
    use_ocr_hyp: bool = False,
    categories: Optional[List[str]] = None,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Buat train + val DataLoader dari metadata_clean.csv.

    Jika tokenizer=None, build otomatis dari data training.

    Contoh:
        train_loader, val_loader = create_dataloaders(
            metadata_path='/ssd-data1/.../metadata_clean.csv',
            images_dir='/ssd-data1/.../images',
            batch_size=128,
        )
    """
    # ── Build tokenizer jika belum ada ────────────────────────────────────────
    if tokenizer is None:
        print("[dataset] Tokenizer tidak diberikan — build vocab dari training data ...")
        # Load semua data training untuk build vocab
        tmp_ds = ITIECSynDataset(
            metadata_path, images_dir=None, tokenizer=None,
            split="train", val_ratio=val_ratio, test_ratio=test_ratio,
            seed=seed, categories=categories, mode="text_only",
        )
        tokenizer = WhitespaceTokenizer()
        tokenizer.build_vocab(tmp_ds.get_text_pairs())
        print(f"[dataset] Vocab size: {len(tokenizer):,} tokens")
        del tmp_ds

    collator = ITIECCollator(pad_id=tokenizer.pad_id)

    train_ds = ITIECSynDataset(
        metadata_path, images_dir, tokenizer,
        split="train", val_ratio=val_ratio, test_ratio=test_ratio,
        seed=seed, categories=categories, mode=mode, use_ocr_hyp=use_ocr_hyp,
    )
    val_ds = ITIECSynDataset(
        metadata_path, images_dir, tokenizer,
        split="val", val_ratio=val_ratio, test_ratio=test_ratio,
        seed=seed, categories=categories, mode=mode, use_ocr_hyp=use_ocr_hyp,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collator,
        pin_memory=pin_memory,
    )

    _print_split_stats(train_ds, val_ds, tokenizer)
    return train_loader, val_loader, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
#  prepare_fairseq_files — Fairseq format untuk CASTLE fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def prepare_fairseq_files(
    metadata_path: str,
    output_dir: str,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    categories: Optional[List[str]] = None,
    use_ocr_hyp: bool = False,
    include_category_tag: bool = True,
) -> Dict[str, str]:
    """
    Hasilkan file plain-text dalam format Fairseq:
        {output_dir}/train.src, train.tgt
        {output_dir}/valid.src, valid.tgt
        {output_dir}/test.src,  test.tgt

    Langkah selanjutnya setelah fungsi ini:
        fairseq-preprocess \\
            --source-lang src --target-lang tgt \\
            --trainpref {output_dir}/train \\
            --validpref {output_dir}/valid \\
            --testpref  {output_dir}/test \\
            --destdir   {output_dir}/../data-bin \\
            --workers   8

    Parameter
    ---------
    include_category_tag : bool
        Jika True, tambahkan tag kategori di awal src (e.g., "<morphological>").
        Berguna untuk multi-task training agar model tahu tipe error.
        Contoh src: "<algospeak> k4mu 5udah makan"

    Return
    ------
    Dict dengan path ke semua file yang dibuat.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_rows = _load_metadata(metadata_path)

    if categories:
        all_rows = [r for r in all_rows if r["category"] in categories]

    splits = {
        "train": _stratified_split(all_rows, "train", val_ratio, test_ratio, seed),
        "valid": _stratified_split(all_rows, "val",   val_ratio, test_ratio, seed),
        "test":  _stratified_split(all_rows, "test",  val_ratio, test_ratio, seed),
    }

    output_files = {}
    total_written = 0

    print(f"\n{'='*60}")
    print(f" ITIEC Fairseq Data Preparation")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f" input  : {metadata_path}")
    print(f" output : {output_dir}")
    print(f" total  : {len(all_rows):,} rows (after category filter)")
    print(f"{'='*60}\n")

    for split_name, rows in splits.items():
        src_path = os.path.join(output_dir, f"{split_name}.src")
        tgt_path = os.path.join(output_dir, f"{split_name}.tgt")

        with open(src_path, "w", encoding="utf-8") as f_src, \
             open(tgt_path, "w", encoding="utf-8") as f_tgt:

            for row in rows:
                # Input text
                if use_ocr_hyp and "ocr_hyp" in row and row["ocr_hyp"]:
                    src_text = row["ocr_hyp"].strip()
                else:
                    src_text = row["src_text"].strip()

                trg_text = row["trg_text"].strip()

                # Optional category tag di awal input
                if include_category_tag:
                    castle_cat = ITIEC_TO_CASTLE.get(row["category"], "morphological")
                    src_text = f"<{castle_cat}> {src_text}"

                f_src.write(src_text + "\n")
                f_tgt.write(trg_text + "\n")

        output_files[f"{split_name}_src"] = src_path
        output_files[f"{split_name}_tgt"] = tgt_path
        total_written += len(rows)

        # Per-split category breakdown
        cat_counts = Counter(r["category"] for r in rows)
        print(f"  {split_name:6s}: {len(rows):>7,} pasangan  "
              f"| {', '.join(f'{c}={n}' for c, n in sorted(cat_counts.items()))}")

    print(f"\n  Total ditulis: {total_written:,} baris")
    print(f"\n  Langkah berikutnya (fairseq-preprocess):")
    print(f"    fairseq-preprocess \\")
    print(f"      --source-lang src --target-lang tgt \\")
    print(f"      --trainpref {output_dir}/train \\")
    print(f"      --validpref {output_dir}/valid \\")
    print(f"      --testpref  {output_dir}/test  \\")
    print(f"      --destdir   {output_dir}/../data-bin \\")
    print(f"      --workers   8\n")

    # Simpan manifest
    manifest = {
        "created":       datetime.now().isoformat(),
        "metadata_path": metadata_path,
        "output_dir":    output_dir,
        "total_pairs":   total_written,
        "splits": {
            k: len(v) for k, v in splits.items()
        },
        "category_tag":  include_category_tag,
        "use_ocr_hyp":   use_ocr_hyp,
        "files":         output_files,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  Manifest: {manifest_path}")

    return output_files


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_metadata(path: str) -> List[Dict]:
    """Load metadata CSV ke list of dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata tidak ditemukan: {path}")
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _stratified_split(
    rows: List[Dict],
    target_split: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> List[Dict]:
    """
    Stratified split per kategori — memastikan distribusi seimbang di semua split.

    Jika CSV sudah punya kolom 'split', pakai langsung.
    Jika tidak, buat split baru berdasarkan val_ratio dan test_ratio.
    """
    # Cek apakah CSV sudah punya kolom 'split' dengan distribusi lengkap
    # (bukan semua 'train' seperti pada ITIEC-Syn synthetic yang belum di-split)
    if rows and "split" in rows[0]:
        split_values = set(r["split"] for r in rows)
        has_real_splits = split_values & {"val", "valid", "test"}
        if has_real_splits:
            # CSV sudah punya split lengkap — pakai langsung
            if target_split == "val":
                return [r for r in rows if r["split"] in ("val", "valid")]
            return [r for r in rows if r["split"] == target_split]
        # Semua 'train' → fall through ke auto-split

    # Auto-split: stratified per kategori
    rng = random.Random(seed)
    by_cat: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "unknown")].append(r)

    train_rows, val_rows, test_rows = [], [], []

    for cat, cat_rows in by_cat.items():
        shuffled = cat_rows[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test  = max(1, int(n_total * test_ratio))
        n_val   = max(1, int(n_total * val_ratio))
        n_train = n_total - n_val - n_test

        train_rows.extend(shuffled[:n_train])
        val_rows.extend(shuffled[n_train:n_train + n_val])
        test_rows.extend(shuffled[n_train + n_val:])

    return {"train": train_rows, "val": val_rows, "test": test_rows}[target_split]


def _pad_sequence(
    sequences: List[torch.Tensor],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad list of 1D tensors → (B, T_max) tensor + mask."""
    max_len = max(s.size(0) for s in sequences)
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    mask   = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, :s.size(0)] = s
        mask[i,   :s.size(0)] = 1
    return padded, mask


def _default_img_transform():
    """Transform standar untuk gambar dalam multimodal mode."""
    return T.Compose([
        T.Resize((_DEFAULT_IMG_SIZE, _DEFAULT_IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _print_split_stats(
    train_ds: ITIECSynDataset,
    val_ds: ITIECSynDataset,
    tokenizer: WhitespaceTokenizer,
):
    print(f"\n{'─'*60}")
    print(f"  Split statistics")
    print(f"{'─'*60}")
    print(f"  Train  : {len(train_ds):>8,} sampel")
    print(f"  Val    : {len(val_ds):>8,} sampel")
    print(f"  Vocab  : {len(tokenizer):>8,} tokens (whitespace)")
    print(f"\n  Train category distribution:")
    for cat, cnt in sorted(train_ds.category_distribution().items()):
        print(f"    {cat:<16} {cnt:>7,}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="ITIEC dataset utility: prepare Fairseq files atau inspect dataset."
    )
    sub = parser.add_subparsers(dest="command")

    # ── prepare: generate Fairseq files ──────────────────────────────────────
    p_prep = sub.add_parser("prepare", help="Generate Fairseq train/valid/test files")
    p_prep.add_argument("--metadata",   required=True, help="Path ke metadata_clean.csv")
    p_prep.add_argument("--out",        required=True, help="Output direktori untuk .src/.tgt files")
    p_prep.add_argument("--val-ratio",  type=float, default=0.05)
    p_prep.add_argument("--test-ratio", type=float, default=0.05)
    p_prep.add_argument("--seed",       type=int,   default=42)
    p_prep.add_argument("--no-cat-tag", action="store_true",
                        help="Jangan tambahkan <category> tag di input")
    p_prep.add_argument("--use-ocr-hyp", action="store_true",
                        help="Gunakan kolom ocr_hyp sebagai input (bukan src_text)")
    p_prep.add_argument("--categories",  nargs="+", default=None,
                        help="Filter kategori (default: semua 6)")

    # ── inspect: tampilkan statistik ──────────────────────────────────────────
    p_insp = sub.add_parser("inspect", help="Tampilkan statistik dataset")
    p_insp.add_argument("--metadata",   required=True, help="Path ke metadata_clean.csv")
    p_insp.add_argument("--val-ratio",  type=float, default=0.05)
    p_insp.add_argument("--test-ratio", type=float, default=0.05)
    p_insp.add_argument("--seed",       type=int,   default=42)

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_fairseq_files(
            metadata_path        = args.metadata,
            output_dir           = args.out,
            val_ratio            = args.val_ratio,
            test_ratio           = args.test_ratio,
            seed                 = args.seed,
            categories           = args.categories,
            use_ocr_hyp          = args.use_ocr_hyp,
            include_category_tag = not args.no_cat_tag,
        )

    elif args.command == "inspect":
        print(f"\nLoading {args.metadata} ...")
        for split in ("train", "val", "test"):
            ds = ITIECSynDataset(
                args.metadata, tokenizer=None, split=split,
                val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                seed=args.seed,
            )
            dist = ds.category_distribution()
            print(f"\n  {split.upper()} — {len(ds):,} sampel")
            for cat, cnt in sorted(dist.items()):
                print(f"    {cat:<16} {cnt:>7,}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
