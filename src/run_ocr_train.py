#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE C / data — OCR PP-OCRv4 pada crop TRAIN+VAL (leakage-free) → pasangan (gt, ocr).
Dipakai utk membangun model konfusi-OCR. Meniru pemanggilan PaddleOCR di ocr_baseline.py.
Output: results/train_ocr_pairs.csv (crop, category, gt, ocr)
"""
import json, os, csv
CROPS = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/raw/crops"
SPLITS = [os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/splits/train.json",
          os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/data/splits/val.json"]
OUT = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+"/results/train_ocr_pairs.csv"

recs = []
for sp in SPLITS:
    for e in json.load(open(sp, encoding="utf-8")):
        base = os.path.splitext(e["filename"])[0]
        for s in e.get("sentences", []):
            cp = os.path.join(CROPS, f"{base}_{s['sentence_id']}.jpg")
            if os.path.exists(cp) and (s.get("teks_ocr") or "").strip():
                recs.append({"crop": cp, "category": e.get("primary_category",""),
                             "gt": s["teks_ocr"].strip()})
print(f"crop train+val: {len(recs)}")

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_textline_orientation=True, lang="en", show_log=False)
def run(cp):
    res = ocr.ocr(cp, cls=False)
    texts = []
    if res and res[0]:
        for line in res[0]:
            if line and len(line) >= 2 and line[1]:
                texts.append(line[1][0])
    return " ".join(texts).strip()

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["crop","category","gt","ocr"])
    for i, r in enumerate(recs):
        w.writerow([r["crop"], r["category"], r["gt"], run(r["crop"])])
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(recs)}")
print(f"→ {OUT}")
