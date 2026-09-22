#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE C / data — generate data latih NOISE-MATCHED (input berisik → target bersih).
Ambil source syn (sudah ber-algospeak) + tambahkan noise-OCR via model konfusi → src berisik.
Target = teks bersih (syn .tgt). Output siap latih corrector tahan-OCR.

  src syn (algospeak) --[OCR noisy channel]--> src_noisy ;  tgt = clean
Opsional: tambah pasangan dari teks bersih murni (IGED) yg dikorupsi algospeak+OCR.

Pakai:
  python gen_noisy_data.py --confusion results/ocr_confusion.json \
     --syn-src data/syn/train.src --syn-tgt data/syn/train.tgt \
     --out-prefix data/noisy/train --reps 1
"""
import argparse, json, random, os, re
random.seed(13)

def load_conf(p):
    m = json.load(open(p, encoding="utf-8"))
    chars = {}
    for c, info in m["chars"].items():
        outs, ws = [], []
        for k, v in info["out"].items():
            outs.append(("DEL" if k == "DEL" else k[1:])); ws.append(v)
        chars[c] = (outs, ws)
    ins_chars = list(m.get("ins_dist", {}).keys()) or [" "]
    ins_w = list(m.get("ins_dist", {}).values()) or [1]
    return chars, m.get("ins_rate", 0.0), ins_chars, ins_w

def ocr_noise(text, chars, ins_rate, ins_chars, ins_w, scale=1.0,
              sub_only=False, keep_space=False):
    """sub_only=True → hanya substitusi (tanpa DEL/INS): jaga panjang & batas kata (realistis).
       keep_space=True → spasi tak pernah diubah/dihapus (jaga batas kata)."""
    out = []
    for c in text:
        if keep_space and c == " ":
            out.append(c); continue
        if c in chars:
            outs, ws = chars[c]
            ch = random.choices(outs, weights=ws, k=1)[0]
            if ch == "DEL":
                if sub_only:
                    out.append(c)                       # tak hapus
                elif random.random() < scale:
                    continue
                else:
                    out.append(c)
            else:
                out.append(c if (ch == c or random.random() > scale) else ch)
        else:
            out.append(c)
        if (not sub_only) and random.random() < ins_rate * scale:
            out.append(random.choices(ins_chars, weights=ins_w, k=1)[0])
    return "".join(out)

STRIP_TAG = re.compile(r"^<[^>]+>\s*")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confusion", required=True)
    ap.add_argument("--syn-src", required=True)   # source syn (ber-algospeak, mungkin ada tag)
    ap.add_argument("--syn-tgt", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--reps", type=int, default=1, help="berapa varian noise per kalimat")
    ap.add_argument("--scale", type=float, default=1.0, help="intensitas noise (0..1)")
    ap.add_argument("--keep-tag", action="store_true")
    ap.add_argument("--sub-only", action="store_true", help="hanya substitusi (realistis, jaga panjang)")
    ap.add_argument("--keep-space", action="store_true", help="jangan ubah/hapus spasi (jaga batas kata)")
    args = ap.parse_args()

    chars, ins_rate, ins_chars, ins_w = load_conf(args.confusion)
    src = [l.rstrip("\n") for l in open(args.syn_src, encoding="utf-8")]
    tgt = [l.rstrip("\n") for l in open(args.syn_tgt, encoding="utf-8")]
    assert len(src) == len(tgt), (len(src), len(tgt))
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    n_out = 0
    with open(args.out_prefix + ".src", "w", encoding="utf-8") as fs, \
         open(args.out_prefix + ".tgt", "w", encoding="utf-8") as ft:
        for s, t in zip(src, tgt):
            tag = ""
            body = s
            m = STRIP_TAG.match(s)
            if m:
                tag = s[:m.end()] if args.keep_tag else ""
                body = s[m.end():]
            for _ in range(args.reps):
                noisy = ocr_noise(body, chars, ins_rate, ins_chars, ins_w, args.scale, args.sub_only, args.keep_space)
                fs.write((tag + noisy).strip() + "\n")
                ft.write(t.strip() + "\n")
                n_out += 1
    print(f"→ {args.out_prefix}.src/.tgt  ({n_out} pasangan)")
    # contoh
    print("Contoh (src berisik → tgt bersih):")
    for s, t in list(zip(src, tgt))[:3]:
        body = STRIP_TAG.sub("", s)
        print("  NOISY:", ocr_noise(body, chars, ins_rate, ins_chars, ins_w, args.scale, args.sub_only, args.keep_space)[:80])
        print("  CLEAN:", t[:80]); print()

if __name__ == "__main__":
    main()
