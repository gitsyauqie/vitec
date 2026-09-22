#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WordPiece-tokenize pasangan src/tgt (tag dipertahankan apa adanya) → .wp.src/.wp.tgt."""
import argparse, os
from tokenizers import Tokenizer
CASTLE = os.environ.get("CASTLE_ROOT", "/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")
TOK = os.environ.get("VITEC_TOKENIZER", f"{CASTLE}/raw-wordpiece/wordpiece_tokenizer/tokenizer.json")
_t=Tokenizer.from_file(TOK)
def wp(s): return " ".join(x for x in _t.encode(s).tokens if x not in ("[CLS]","[SEP]"))
ap=argparse.ArgumentParser()
ap.add_argument("--src",required=True); ap.add_argument("--tgt",required=True)
ap.add_argument("--out-prefix",required=True)
a=ap.parse_args()
S=[l.rstrip("\n") for l in open(a.src,encoding="utf-8")]
T=[l.rstrip("\n") for l in open(a.tgt,encoding="utf-8")]
assert len(S)==len(T),(len(S),len(T))
with open(a.out_prefix+".wp.src","w",encoding="utf-8") as fs, \
     open(a.out_prefix+".wp.tgt","w",encoding="utf-8") as ft:
    for s,t in zip(S,T):
        fs.write(wp(s)+"\n"); ft.write(wp(t)+"\n")
print(f"→ {a.out_prefix}.wp.src/.wp.tgt ({len(S)} baris)")
