#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Siapkan test set C3 = OCR→L2-normalized → (akan di-GEC).
Baca results/e2e_ocr_norm.csv (kolom 'normalized'), WordPiece-tokenize dgn tag benar,
tulis test_c3.wp.src/.tgt + meta — siap utk preprocess + gen_eval_da.py.
"""
import csv, json, os
ROOT=os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+""
OUT=f"{ROOT}/data/real_da_wp"
NORM_CSV=f"{ROOT}/results/e2e_ocr_norm.csv"
TOKENIZER=os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/raw-wordpiece/wordpiece_tokenizer/tokenizer.json"
TAG_MAP={"ALG":"<semantic>","SEM":"<semantic>","SIN":"<syntactic>",
         "EJA":"<morphological>","MOR":"<morphological>","TIP":"<morphological>","MIX":"<morphological>"}
DEFAULT="<morphological>"
from tokenizers import Tokenizer
_t=Tokenizer.from_file(TOKENIZER)
def wp(s): return " ".join(x for x in _t.encode(s).tokens if x not in ("[CLS]","[SEP]"))

rows=list(csv.DictReader(open(NORM_CSV,encoding="utf-8")))
src,tgt,meta=[],[],[]
for i,r in enumerate(rows):
    cat=(r.get("category") or "").upper()
    tag=TAG_MAP.get(cat,DEFAULT)
    s=(r.get("normalized") or "").strip()
    ref=(r.get("kalimat_koreksi") or "").strip()
    src.append(wp(f"{tag} {s}")); tgt.append(wp(ref))
    meta.append({"id":i,"category":cat,"src_orig":s,"ref_orig":ref})
os.makedirs(OUT,exist_ok=True)
open(f"{OUT}/test_c3.wp.src","w",encoding="utf-8").write("\n".join(src)+"\n")
open(f"{OUT}/test_c3.wp.tgt","w",encoding="utf-8").write("\n".join(tgt)+"\n")
with open(f"{OUT}/test_c3.meta.jsonl","w",encoding="utf-8") as f:
    for m in meta: f.write(json.dumps(m,ensure_ascii=False)+"\n")
print(f"→ {OUT}/test_c3.wp.{{src,tgt}} + meta ({len(src)} baris)")
