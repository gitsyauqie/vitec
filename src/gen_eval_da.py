#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX T6 — generate GEC pada test set WordPiece, detok, tulis CSV siap-eval.

Pakai (per konfigurasi tangga):
  # C2 (oracle transcription → GEC)
  python gen_eval_da.py --bin data/processed/da_test_c2_bin \
      --meta data/real_da_wp/test_c2.meta.jsonl \
      --ckpt models/checkpoints/castle_da_nokg_wp/checkpoint_best.pt \
      --out results/e2e_C2.csv --gpu 4

  # C1 (real OCR → GEC)
  python gen_eval_da.py --bin data/processed/da_test_c1_bin \
      --meta data/real_da_wp/test_c1.meta.jsonl \
      --ckpt models/checkpoints/castle_da_nokg_wp/checkpoint_best.pt \
      --out results/e2e_C1.csv --gpu 4

Lalu eval (protokol terdokumentasi):
  python scripts/eval_protocol_v2.py --pred results/e2e_C2.csv --run-id C2 \
     --layer L3 E2E --cer-norm lower \
     --col-cat category --col-src source --col-hyp hyp --col-ref kalimat_koreksi \
     --out results/tables/ladder_ccroplevel.csv
"""
import argparse, json, os, subprocess, sys

EXT = os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/fairseq_extensions"
KG  = os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/semantic_kg.json"

import re as _re
def _fix_punct(s: str) -> str:
    s = _re.sub(r"\s+([,.!?;:%)\]\}»”’])", r"\1", s)   # spasi sebelum tanda baca penutup
    s = _re.sub(r"([(\[\{«“‘])\s+", r"\1", s)            # spasi setelah pembuka
    s = _re.sub(r"\s+([*'\"-])\s+", r"\1", s)            # rapatkan * ' \" - di tengah (as * sila→as*sila)
    s = _re.sub(r"\s{2,}", " ", s)
    return s.strip()

def detok_wp(s: str) -> str:
    """WordPiece detok: '##x' nempel ke token sebelumnya + rapatkan tanda baca."""
    out = []
    for t in s.split():
        if t in ("[CLS]", "[SEP]"):
            continue
        if t.startswith("##"):
            if out: out[-1] += t[2:]
            else:   out.append(t[2:])
        else:
            out.append(t)
    return _fix_punct(" ".join(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ext", default=EXT)
    ap.add_argument("--kg", default=KG)
    ap.add_argument("--gpu", default="4")
    ap.add_argument("--beam", default="5")
    ap.add_argument("--subset", default="test", help="gen-subset (test|valid|train)")
    ap.add_argument("--src-lang", default="src", help="src lang name di bin (src|source)")
    ap.add_argument("--tgt-lang", default="tgt", help="tgt lang name di bin (tgt|target)")
    args = ap.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    cmd = [
        "fairseq-generate", args.bin,
        "--user-dir", args.ext,
        "--path", args.ckpt,
        "--task", "translation",
        "--source-lang", args.src_lang, "--target-lang", args.tgt_lang,
        "--model-overrides", json.dumps({"kg_path": args.kg}),
        "--beam", args.beam,
        "--max-tokens", "4096",
        "--gen-subset", args.subset,
    ]
    print("RUN:", " ".join(cmd))
    gen = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if gen.returncode != 0:
        sys.stderr.write(gen.stderr[-3000:])
        raise SystemExit("fairseq-generate gagal")

    hyps, scores = {}, {}
    for line in gen.stdout.splitlines():
        if line.startswith("H-"):
            p = line.split("\t")
            if len(p) >= 3:
                idx = int(p[0].split("-")[1])
                scores[idx] = float(p[1])          # avg log-prob hipotesis = confidence
                hyps[idx] = detok_wp(p[2])

    meta = [json.loads(l) for l in open(args.meta, encoding="utf-8") if l.strip()]
    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_miss = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "source", "hyp", "kalimat_koreksi", "gec_score"])
        for m in meta:
            i = m["id"]
            hyp = hyps.get(i)
            if hyp is None:
                n_miss += 1
                hyp = m["src_orig"]   # fallback: tak ada output → pakai source
            w.writerow([i, m["category"], m["src_orig"], hyp, m["ref_orig"], scores.get(i, "")])
    print(f"→ {args.out}  ({len(meta)} baris, {n_miss} tanpa hyp)")
    # contoh untuk inspeksi mata
    for m in meta[:3]:
        print("  SRC:", m["src_orig"][:80])
        print("  HYP:", hyps.get(m["id"], "(none)")[:80])
        print("  REF:", m["ref_orig"][:80]); print()

if __name__ == "__main__":
    main()
