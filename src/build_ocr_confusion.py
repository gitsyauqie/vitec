#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE C / data — bangun model konfusi-OCR (noisy channel) dari pasangan (gt, ocr).
Align char-level (difflib) → distribusi per-karakter: keep / substitusi→x / delete,
plus laju & distribusi insertion. Disimpan JSON utk gen_noisy_data.py.

Pakai:
  python build_ocr_confusion.py --pairs results/train_ocr_pairs.csv --out results/ocr_confusion.json
  # fallback bila OCR train belum ada (memodelkan engine; catat di paper):
  python build_ocr_confusion.py --pairs results/e2e_ocr_outputs.csv --gt-col teks_ocr --ocr-col paddleocr_out --out results/ocr_confusion.json
"""
import argparse, csv, json
from collections import defaultdict, Counter
from difflib import SequenceMatcher

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--gt-col", default="gt")
    ap.add_argument("--ocr-col", default="ocr")
    ap.add_argument("--out", default="results/ocr_confusion.json")
    args = ap.parse_args()

    sub = defaultdict(Counter)   # true_char → Counter(ocr_char)  (termasuk keep: c→c)
    dele = Counter()             # true_char → jml dihapus
    ins = Counter()              # ocr_char disisipkan
    n_true = Counter()           # total kemunculan true_char
    n_pairs = n_inspos = 0

    rows = list(csv.DictReader(open(args.pairs, encoding="utf-8")))
    for r in rows:
        gt = (r.get(args.gt_col) or ""); oc = (r.get(args.ocr_col) or "")
        if not gt: continue
        n_pairs += 1
        sm = SequenceMatcher(a=list(gt), b=list(oc), autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for c in gt[i1:i2]:
                    sub[c][c] += 1; n_true[c] += 1
            elif tag == "replace":
                g, o = gt[i1:i2], oc[j1:j2]
                m = min(len(g), len(o))
                for k in range(m):
                    sub[g[k]][o[k]] += 1; n_true[g[k]] += 1
                for c in g[m:]:                # sisa gt → delete
                    dele[c] += 1; n_true[c] += 1
                for c in o[m:]:                # sisa ocr → insert
                    ins[c] += 1
            elif tag == "delete":
                for c in gt[i1:i2]:
                    dele[c] += 1; n_true[c] += 1
            elif tag == "insert":
                for c in oc[j1:j2]:
                    ins[c] += 1
        n_inspos += max(len(gt), 1)

    # bangun distribusi per true_char: P(keep/sub/del)
    model = {"chars": {}, "ins_rate": sum(ins.values())/max(n_inspos,1),
             "ins_dist": dict(ins.most_common(40)), "n_pairs": n_pairs}
    for c, tot in n_true.items():
        d = dict(sub[c]); d_del = dele.get(c, 0)
        outcomes = {f"={k}": v for k, v in d.items()}      # '=x' = jadi x (x==c berarti keep)
        if d_del: outcomes["DEL"] = d_del
        model["chars"][c] = {"total": tot, "out": outcomes}

    json.dump(model, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # ringkas: char paling sering "salah"
    err = []
    for c, info in model["chars"].items():
        keep = info["out"].get(f"={c}", 0); tot = info["total"]
        if tot >= 20:
            err.append((1 - keep/tot, c, tot))
    err.sort(reverse=True)
    print(f"pairs={n_pairs} | ins_rate={model['ins_rate']:.4f} | chars={len(model['chars'])}")
    print("Top karakter rawan-error (err-rate, char, n):")
    for e, c, t in err[:15]:
        top = Counter({k[1:]: v for k, v in model["chars"][c]["out"].items() if not k.startswith("DEL")})
        alt = [f"{k}:{v}" for k, v in top.most_common(4) if k != c]
        print(f"  {e*100:5.1f}%  {repr(c):6s} n={t:5d}  → {', '.join(alt[:3]) or '(del)'}")
    print(f"→ {args.out}")

if __name__ == "__main__":
    main()
