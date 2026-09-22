#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VITEC RQ5 — Fase A: Protokol Evaluasi Berlapis (layered evaluation)
==================================================================

Tujuan: berhenti memvonis pipeline dengan SATU angka CER E2E.
Skrip ini menghitung metrik yang BENAR untuk tiap layer + bootstrap CI,
lalu mengisi sel tangga cascade-oracle (C0..C6).

Metrik per layer
----------------
  L1 (OCR)            : CER & WER  ->  OCR-output vs TRANSKRIPSI ASLI (teks dengan error),
                                       BUKAN vs kalimat koreksi. (isolasi recognition)
  L2 (Normalisasi)    : token-level Precision/Recall/F1 pada token yang HARUS dinormalisasi
                                       (correction-as-detection: source!=gold).
  L3 (GEC)            : edit-level F0.5 (Max-Match style) source->hyp vs source->ref.
                                       Punct-spacing normalization opsional (default ON),
                                       konsisten dgn "normalized F0.5 = 66.49%" di plan.
  E2E (CER)           : CER output-akhir vs kalimat_koreksi (untuk tangga oracle).

Semua metrik corpus-level + bootstrap 95% CI (resample kalimat) + breakdown per kategori.

CARA PAKAI (di server)
----------------------
  # 1) Verifikasi metrik benar dulu (sintetis, tanpa data):
  python eval_protocol_v2.py --selftest

  # 2) Jalankan pada satu file prediksi (satu konfigurasi tangga, mis. C1):
  python eval_protocol_v2.py \
      --pred  results/e2e/C1_real_ocr_gec.csv \
      --run-id C1 \
      --layer L3 E2E \
      --out   results/tables/eval_protocol_v2.csv

FORMAT INPUT (CSV/TSV/JSONL) — sesuaikan nama kolom via flag bila beda
---------------------------------------------------------------------
  id        : id kalimat/region (untuk join & bootstrap)        [--col-id]
  category  : ALG/EJA/MOR/SEM/TIP/...                            [--col-cat]
  source    : teks INPUT ke tahap yg dievaluasi                 [--col-src]
              - L1: (tidak dipakai)
              - L2: teks sebelum normalisasi (OCR mentah)
              - L3: teks ternormalisasi (input ke GEC)
  hyp       : OUTPUT sistem pada layer tsb                       [--col-hyp]
              - L1: teks_ocr  | L2: hasil normalizer | L3: output GEC | E2E: output akhir
  ref       : GOLD reference                                     [--col-ref]
              - L1: transkripsi asli (teks dgn error)            [--col-ref-l1]
              - L2: bentuk ternormalisasi gold                   [--col-ref-l2]
              - L3 & E2E: kalimat_koreksi                        [--col-ref]

Tidak ada dependency eksternal (pure Python) -> aman di env fairseq lama.
"""
import argparse, csv, json, os, sys, random, re
from collections import defaultdict

# --------------------------------------------------------------------------- #
#  Edit distance primitives (pure python, no deps)                            #
# --------------------------------------------------------------------------- #
def _lev(a, b):
    """Levenshtein distance over sequences a, b (lists or strings)."""
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]

def cer(ref, hyp):
    ref = ref or ""
    if len(ref) == 0:
        return 0.0 if len(hyp or "") == 0 else 1.0
    return _lev(list(ref), list(hyp or "")) / len(ref)

def wer(ref, hyp):
    r = (ref or "").split()
    h = (hyp or "").split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _lev(r, h) / len(r)

# --------------------------------------------------------------------------- #
#  Tokenisation / normalisation helpers                                       #
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"\s+([,.!?;:%)\]\}])|([(\[\{])\s+")
def normalize_punct_spacing(s):
    """Hilangkan spasi artefak WordPiece di sekitar tanda baca."""
    if not s: return s
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    s = re.sub(r"([(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+([)\]\}])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def simple_tok(s):
    if not s: return []
    return s.split()

# Normalisasi untuk CER (di-set dari CLI). CER sangat sensitif thd kapitalisasi & tanda baca,
# jadi protokol HARUS eksplisit. Default: raw (apa adanya).
CER_NORM = "raw"   # "raw" | "lower" | "lower_nopunct"
def normalize_for_cer(s):
    if not s: return s or ""
    if CER_NORM == "raw":
        return s
    s = s.lower()
    if CER_NORM == "lower_nopunct":
        s = re.sub(r"[^\w\s]", "", s)        # buang semua tanda baca
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --------------------------------------------------------------------------- #
#  Edit extraction (source -> target) via difflib                             #
# --------------------------------------------------------------------------- #
from difflib import SequenceMatcher
def extract_edits(src_toks, tgt_toks):
    """
    Himpunan edit (i, j, 'src_span', 'tgt_span') dari src->tgt.
    Memakai opcode SequenceMatcher; tiap replace/insert/delete = satu edit.
    Edit direpresentasikan sebagai tuple posisi+isi agar bisa di-set-interset.
    """
    sm = SequenceMatcher(a=src_toks, b=tgt_toks, autojunk=False)
    edits = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        edits.add((i1, i2, " ".join(src_toks[i1:i2]), " ".join(tgt_toks[j1:j2])))
    return edits

def fbeta(tp, fp, fn, beta=0.5):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p == 0 and r == 0:
        return 0.0, 0.0, 0.0
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return p, r, f

# --------------------------------------------------------------------------- #
#  Per-sentence raw counts (supaya bootstrap = resample lalu agregasi)        #
# --------------------------------------------------------------------------- #
def l1_counts(row):
    """CER/WER butuh (edit_chars, ref_chars) & (edit_words, ref_words)."""
    ref = normalize_for_cer(row["ref_l1"]); hyp = normalize_for_cer(row["hyp"])
    rc = list(ref or ""); hc = list(hyp or "")
    rw = (ref or "").split(); hw = (hyp or "").split()
    return {
        "char_edits": _lev(rc, hc), "char_ref": max(len(rc), 1) if ref else 0,
        "word_edits": _lev(rw, hw), "word_ref": len(rw),
    }

def l2_counts(row):
    """
    correction-as-detection token-level.
    Butuh src (pra-norm), hyp (hasil norm), ref_l2 (gold-norm), token-aligned by index.
    TP: token yg HARUS diubah (src!=ref) DAN hyp==ref
    FP: token yg diubah (hyp!=src) DAN hyp!=ref
    FN: token yg HARUS diubah (src!=ref) DAN hyp!=ref
    """
    src = simple_tok(row["src"]); hyp = simple_tok(row["hyp"]); ref = simple_tok(row["ref_l2"])
    n = max(len(src), len(hyp), len(ref))
    src += [""] * (n - len(src)); hyp += [""] * (n - len(hyp)); ref += [""] * (n - len(ref))
    tp = fp = fn = 0
    for s, h, g in zip(src, hyp, ref):
        need = (s != g)
        changed = (h != s)
        if need and h == g: tp += 1
        elif need and h != g: fn += 1
        elif (not need) and changed and h != g: fp += 1
    return {"tp": tp, "fp": fp, "fn": fn}

def l3_counts(row, norm_punct=True):
    """edit-level F0.5: bandingkan himpunan edit src->hyp vs src->ref.
       Case-insensitive (model uncased) — bandingkan dalam lowercase."""
    src, hyp, ref = row["src"].lower(), row["hyp"].lower(), row["ref"].lower()
    if norm_punct:
        src = normalize_punct_spacing(src); hyp = normalize_punct_spacing(hyp); ref = normalize_punct_spacing(ref)
    st = simple_tok(src); ht = simple_tok(hyp); rt = simple_tok(ref)
    gold = extract_edits(st, rt)
    sysd = extract_edits(st, ht)
    tp = len(gold & sysd); fp = len(sysd - gold); fn = len(gold - sysd)
    return {"tp": tp, "fp": fp, "fn": fn}

def e2e_counts(row):
    ref = normalize_for_cer(row["ref"]); hyp = normalize_for_cer(row["hyp"])
    rc = list(ref or "")
    return {"char_edits": _lev(rc, list(hyp or "")), "char_ref": len(rc)}

# --------------------------------------------------------------------------- #
#  Aggregation                                                                #
# --------------------------------------------------------------------------- #
def agg_metric(layer, counts_list, norm_punct=True):
    if layer == "L1":
        ce = sum(c["char_edits"] for c in counts_list); cr = sum(c["char_ref"] for c in counts_list)
        we = sum(c["word_edits"] for c in counts_list); wr = sum(c["word_ref"] for c in counts_list)
        return {"CER": ce / cr if cr else 0.0, "WER": we / wr if wr else 0.0}
    if layer == "E2E":
        ce = sum(c["char_edits"] for c in counts_list); cr = sum(c["char_ref"] for c in counts_list)
        return {"CER": ce / cr if cr else 0.0}
    # L2 / L3 -> P/R/F0.5
    tp = sum(c["tp"] for c in counts_list); fp = sum(c["fp"] for c in counts_list); fn = sum(c["fn"] for c in counts_list)
    p, r, f = fbeta(tp, fp, fn, beta=0.5)
    return {"P": p, "R": r, "F0.5": f, "tp": tp, "fp": fp, "fn": fn}

def bootstrap_ci(layer, counts_list, key, n_boot=1000, seed=42):
    if len(counts_list) < 2:
        return (None, None)
    rng = random.Random(seed)
    vals = []
    idx = range(len(counts_list))
    for _ in range(n_boot):
        sample = [counts_list[rng.randrange(len(counts_list))] for _ in idx]
        m = agg_metric(layer, sample)
        if key in m:
            vals.append(m[key])
    if not vals: return (None, None)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]; hi = vals[int(0.975 * len(vals)) - 1]
    return (lo, hi)

# --------------------------------------------------------------------------- #
#  IO                                                                         #
# --------------------------------------------------------------------------- #
def load_rows(path, cols):
    rows = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(l) for l in f if l.strip()]
    elif ext == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # bisa berupa array, atau dict berisi array (cari list of dict pertama)
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), data)
        recs = data if isinstance(data, list) else [data]
    else:
        delim = "\t" if ext in (".tsv", ".txt") else ","
        with open(path, encoding="utf-8") as f:
            recs = list(csv.DictReader(f, delimiter=delim))
    for rec in recs:
        rows.append({
            "id":   rec.get(cols["id"], ""),
            "cat":  rec.get(cols["cat"], "ALL"),
            "src":  rec.get(cols["src"], "") or "",
            "hyp":  rec.get(cols["hyp"], "") or "",
            "ref":  rec.get(cols["ref"], "") or "",
            "ref_l1": rec.get(cols["ref_l1"], rec.get(cols["ref"], "")) or "",
            "ref_l2": rec.get(cols["ref_l2"], rec.get(cols["ref"], "")) or "",
        })
    return rows

COUNT_FN = {"L1": l1_counts, "L2": l2_counts, "L3": l3_counts, "E2E": e2e_counts}
PRIMARY_KEY = {"L1": "CER", "L2": "F0.5", "L3": "F0.5", "E2E": "CER"}

def evaluate(rows, layer, norm_punct=True, n_boot=1000):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["cat"]].append(r)
    groups["ALL"] = rows
    for cat, rs in groups.items():
        cfn = COUNT_FN[layer]
        cl = [cfn(r, norm_punct) if layer == "L3" else cfn(r) for r in rs]
        m = agg_metric(layer, cl, norm_punct)
        key = PRIMARY_KEY[layer]
        lo, hi = bootstrap_ci(layer, cl, key, n_boot=n_boot)
        row = {"layer": layer, "category": cat, "n": len(rs)}
        row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
        row["ci95_low"] = round(lo, 4) if lo is not None else ""
        row["ci95_high"] = round(hi, 4) if hi is not None else ""
        out.append(row)
    # urutkan: ALL terakhir
    out.sort(key=lambda x: (x["category"] == "ALL", x["category"]))
    return out

def write_csv(path, run_id, results):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["run_id", "layer", "category", "n", "CER", "WER", "P", "R", "F0.5",
              "tp", "fp", "fn", "ci95_low", "ci95_high"]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header: w.writeheader()
        for r in results:
            r2 = {"run_id": run_id}; r2.update(r)
            w.writerow({k: r2.get(k, "") for k in fields})

# --------------------------------------------------------------------------- #
#  SELF TEST (verifikasi metrik benar tanpa data nyata)                       #
# --------------------------------------------------------------------------- #
def selftest():
    print("=== SELF-TEST eval_protocol_v2 ===\n")
    ok = True

    # --- CER/WER sanity ---
    assert abs(cer("kepala", "kepala")) < 1e-9
    assert abs(cer("kepala", "kepaIa") - 1/6) < 1e-9, cer("kepala","kepaIa")
    assert abs(wer("saya makan nasi", "saya makan roti") - 1/3) < 1e-9
    print("[OK] CER/WER dasar benar (1 subst / panjang ref)")

    # --- L1 agregasi corpus-level ---
    rows = [
        {"ref_l1": "kepala", "hyp": "kepaIa"},   # 1/6
        {"ref_l1": "mayat",  "hyp": "mayat"},    # 0/5
    ]
    cl = [l1_counts(r) for r in rows]
    m = agg_metric("L1", cl)
    exp = 1 / (6 + 5)
    assert abs(m["CER"] - exp) < 1e-9, (m["CER"], exp)
    print(f"[OK] L1 CER corpus-level = {m['CER']:.4f} (expected {exp:.4f})")

    # --- L2 detection (algospeak normalization) ---
    # src: 'k3p4l4 mayat di jalan'  gold-norm: 'kepala mayat di jalan'
    # hyp-A (sempurna): 'kepala mayat di jalan'  -> TP=1,FP=0,FN=0 -> F1=1
    rA = [{"src": "k3p4l4 mayat di jalan", "hyp": "kepala mayat di jalan", "ref_l2": "kepala mayat di jalan"}]
    mA = agg_metric("L2", [l2_counts(r) for r in rA])
    assert mA["tp"] == 1 and mA["fp"] == 0 and mA["fn"] == 0, mA
    assert abs(mA["F0.5"] - 1.0) < 1e-9
    print(f"[OK] L2 normalizer sempurna -> F0.5=1.0 (tp=1,fp=0,fn=0)")
    # hyp-B (over-correct 'mayat'->'sayat', miss 'k3p4l4'): TP=0, FP=1, FN=1
    rB = [{"src": "k3p4l4 mayat di jalan", "hyp": "k3p4l4 sayat di jalan", "ref_l2": "kepala mayat di jalan"}]
    mB = agg_metric("L2", [l2_counts(r) for r in rB])
    assert mB["tp"] == 0 and mB["fp"] == 1 and mB["fn"] == 1, mB
    print(f"[OK] L2 over-correct+miss -> tp=0,fp=1,fn=1, F0.5={mB['F0.5']:.3f}")

    # --- L3 edit-level F0.5 ---
    # src='saya makan nasi goreng'  ref='saya memakan nasi goreng' (1 gold edit: makan->memakan)
    # hyp sempurna -> TP=1 FP=0 FN=0
    r3 = [{"src": "saya makan nasi goreng", "hyp": "saya memakan nasi goreng",
           "ref": "saya memakan nasi goreng"}]
    m3 = agg_metric("L3", [l3_counts(r) for r in r3])
    assert m3["tp"] == 1 and m3["fp"] == 0 and m3["fn"] == 0, m3
    print(f"[OK] L3 koreksi tepat -> F0.5={m3['F0.5']:.3f} (tp=1)")
    # hyp hallucinate proper noun: src=ref=no-edit, hyp ubah 'grobogan'->'grobagi' -> FP=1
    r3b = [{"src": "rumah di grobogan", "hyp": "rumah di grobagi", "ref": "rumah di grobogan"}]
    m3b = agg_metric("L3", [l3_counts(r) for r in r3b])
    assert m3b["tp"] == 0 and m3b["fp"] == 1 and m3b["fn"] == 0, m3b
    print(f"[OK] L3 halusinasi entitas dihukum -> fp=1, F0.5={m3b['F0.5']:.3f}")

    # --- L3 precision-weighting (F0.5 menekankan precision) ---
    # 1 gold edit, sistem benar 1 tapi tambah 1 FP -> P=0.5 R=1 -> F0.5 < F1
    p, r, f = fbeta(1, 1, 0, 0.5)
    assert abs(p - 0.5) < 1e-9 and abs(r - 1.0) < 1e-9
    assert f < (2 * p * r / (p + r)), "F0.5 harus < F1 saat precision rendah"
    print(f"[OK] F0.5 menekankan precision: P=.5 R=1 -> F0.5={f:.3f} (< F1)")

    # --- E2E CER: GEC merusak vs OCR-only ---
    ocr  = [{"ref": "tutup kepala anda", "hyp": "tutup k3pala anda"}]       # OCR-only
    gec  = [{"ref": "tutup kepala anda", "hyp": "tutup nintendo representative"}]  # GEC halusinasi
    cer_ocr = agg_metric("E2E", [e2e_counts(r) for r in ocr])["CER"]
    cer_gec = agg_metric("E2E", [e2e_counts(r) for r in gec])["CER"]
    assert cer_gec > cer_ocr, (cer_gec, cer_ocr)
    print(f"[OK] E2E mereproduksi pola 'GEC merusak': CER OCR={cer_ocr:.3f} < CER+GEC={cer_gec:.3f}")

    # --- bootstrap CI mengandung point estimate ---
    many = [{"ref_l1": "kepala", "hyp": "kepaIa"} for _ in range(20)] + \
           [{"ref_l1": "mayat", "hyp": "mayat"} for _ in range(20)]
    cl = [l1_counts(r) for r in many]
    point = agg_metric("L1", cl)["CER"]
    lo, hi = bootstrap_ci("L1", cl, "CER", n_boot=500)
    assert lo <= point <= hi, (lo, point, hi)
    print(f"[OK] bootstrap CI [{lo:.3f},{hi:.3f}] memuat point={point:.3f}")

    print("\n=== SEMUA SELF-TEST LULUS ===")
    return ok

# --------------------------------------------------------------------------- #
#  CLI                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="VITEC RQ5 Fase A — layered eval")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--layer", nargs="+", default=["L3", "E2E"],
                    choices=["L1", "L2", "L3", "E2E"])
    ap.add_argument("--out", default="results/tables/eval_protocol_v2.csv")
    ap.add_argument("--no-norm-punct", action="store_true", help="matikan normalisasi spasi tanda baca (L3)")
    ap.add_argument("--cer-norm", choices=["raw", "lower", "lower_nopunct"], default="raw",
                    help="normalisasi sebelum CER (L1/E2E): raw | lower | lower_nopunct")
    ap.add_argument("--n-boot", type=int, default=1000)
    # nama kolom (sesuaikan ke file kamu)
    ap.add_argument("--col-id", default="id")
    ap.add_argument("--col-cat", default="category")
    ap.add_argument("--col-src", default="source")
    ap.add_argument("--col-hyp", default="hyp")
    ap.add_argument("--col-ref", default="kalimat_koreksi")
    ap.add_argument("--col-ref-l1", default="teks_asli")
    ap.add_argument("--col-ref-l2", default="teks_ternormalisasi")
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not args.pred:
        ap.error("--pred wajib (atau pakai --selftest)")

    global CER_NORM
    CER_NORM = args.cer_norm

    cols = {"id": args.col_id, "cat": args.col_cat, "src": args.col_src,
            "hyp": args.col_hyp, "ref": args.col_ref,
            "ref_l1": args.col_ref_l1, "ref_l2": args.col_ref_l2}
    rows = load_rows(args.pred, cols)
    print(f"Loaded {len(rows)} baris dari {args.pred} | cer_norm={CER_NORM}")
    for layer in args.layer:
        res = evaluate(rows, layer, norm_punct=not args.no_norm_punct, n_boot=args.n_boot)
        write_csv(args.out, args.run_id, res)
        allrow = next(r for r in res if r["category"] == "ALL")
        key = PRIMARY_KEY[layer]
        print(f"[{args.run_id}] {layer:3s} ALL {key}={allrow.get(key)} "
              f"CI95=[{allrow['ci95_low']},{allrow['ci95_high']}] (n={allrow['n']})")
    print(f"-> ditulis ke {args.out}")

if __name__ == "__main__":
    main()
