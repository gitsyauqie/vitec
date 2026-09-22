#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Item-2: meta utk eval syn-valid (KG ablation). data/syn/valid.{src,tgt} → meta jsonl
   (id, category dari tag, src_orig=teks tanpa tag, ref_orig=tgt). Urut = urutan bin valid."""
import json, re, os
ROOT=os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")
S=[l.rstrip("\n") for l in open(f"{ROOT}/data/syn/valid.src",encoding="utf-8")]
T=[l.rstrip("\n") for l in open(f"{ROOT}/data/syn/valid.tgt",encoding="utf-8")]
assert len(S)==len(T),(len(S),len(T))
TAG=re.compile(r"^<([^>]+)>\s*")
out=f"{ROOT}/data/real_da_wp/synval.meta.jsonl"
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w",encoding="utf-8") as f:
    for i,(s,t) in enumerate(zip(S,T)):
        m=TAG.match(s); cat=(m.group(1) if m else "").upper()[:3]
        body=TAG.sub("",s)
        f.write(json.dumps({"id":i,"category":cat,"src_orig":body.strip(),"ref_orig":t.strip()},ensure_ascii=False)+"\n")
print(f"→ {out} ({len(S)} baris)")
