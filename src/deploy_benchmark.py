#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE D — Deployability benchmark (server CPU, single-thread = proxy ARM).

Ukur untuk corrector CASTLE (dan L2 normalizer):
  1) ukuran model: file checkpoint, state_dict FP32, state_dict INT8 (dynamic quant)
  2) latency/kalimat (batch=1, beam 5, CPU 1 thread): FP32 vs INT8 (median, p95)
  3) peak RAM (ru_maxrss)
  4) degradasi akurasi FP32→INT8: tulis hyp CSV keduanya → eval CER dgn
     eval_protocol_v2 (dibandingkan di luar skrip ini)
  5) (opsional --l2-csv) latency L2 normalizer per kalimat

Pakai (server, env fairseq — CPU saja, tak perlu GPU):
  python deploy_benchmark.py \
    --ckpt ../models/checkpoints/castle_noise_matched/checkpoint_best.pt \
    --bin  ../data/processed/da_test_c3_bin \
    --meta ../data/real_da_wp/test_c3.meta.jsonl \
    --l2-csv ../results/e2e_ocr_outputs.csv \
    --out-prefix ../results/deploy

Lalu eval degradasi:
  python eval_protocol_v2.py --pred ../results/deploy_fp32.csv --run-id DEPLOY_FP32 \
     --layer E2E --cer-norm lower_nopunct --col-cat category --col-src source \
     --col-hyp hyp --col-ref kalimat_koreksi --out ../results/tables/deploy.csv
  (ulangi utk deploy_int8.csv)
"""
import argparse, csv, json, os, re, resource, statistics as st, sys, tempfile, time

# ── util ─────────────────────────────────────────────────────────────────────
def mb(nbytes): return nbytes / (1024 * 1024)

def peak_ram_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # linux: KB

_PUNCT_FIX = [
    (re.compile(r"\s+([,.!?;:%)\]\}»”’])"), r"\1"),
    (re.compile(r"([(\[\{«“‘])\s+"), r"\1"),
    (re.compile(r"\s+([*'\"-])\s+"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
]
def detok_wp(s):
    out = []
    for t in s.split():
        if t in ("[CLS]", "[SEP]"): continue
        if t.startswith("##"):
            if out: out[-1] += t[2:]
            else: out.append(t[2:])
        else: out.append(t)
    s = " ".join(out)
    for rx, rep in _PUNCT_FIX: s = rx.sub(rep, s)
    return s.strip()

def state_dict_size_mb(model):
    import torch
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return mb(os.path.getsize(f.name))

def summarize(times):
    ts = sorted(times)
    return dict(n=len(ts), mean=st.mean(ts), median=ts[len(ts)//2],
                p95=ts[int(0.95*len(ts))], total=sum(ts))

# ── benchmark inti ───────────────────────────────────────────────────────────
def build_generator(task, models, beam):
    import argparse as ap
    g = ap.Namespace(beam=beam, max_len_a=1.2, max_len_b=10, min_len=1,
                     nbest=1, lenpen=1.0, unkpen=0.0, temperature=1.0,
                     match_source_len=False, no_repeat_ngram_size=0,
                     sampling=False, sampling_topk=-1, sampling_topp=-1.0,
                     diverse_beam_groups=-1, diverse_beam_strength=0.5,
                     diversity_rate=-1.0, constraints=None, prefix_size=0,
                     replace_unk=None, score_reference=False, retain_dropout=False)
    try:
        return task.build_generator(models, g)
    except TypeError:
        return task.build_generator(g)   # fairseq lama

def run_model(tag, models, task, subset, beam, limit, warmup=3):
    """generate per-kalimat (batch=1); return (times, {id: (hyp_wp, score)})."""
    import torch
    generator = build_generator(task, models, beam)
    itr = task.get_batch_iterator(
        dataset=task.dataset(subset), max_sentences=1,
        ignore_invalid_inputs=True, required_batch_size_multiple=1,
    ).next_epoch_itr(shuffle=False)
    times, hyps = [], {}
    tgt_dict = task.target_dictionary
    with torch.no_grad():
        for i, sample in enumerate(itr):
            if limit and i >= limit + warmup: break
            t0 = time.perf_counter()
            out = task.inference_step(generator, models, sample)
            dt = time.perf_counter() - t0
            if i >= warmup: times.append(dt)
            sid = int(sample["id"][0])
            best = out[0][0]
            hyp_wp = tgt_dict.string(best["tokens"].int().cpu())
            score = float(best["score"])
            hyps[sid] = (detok_wp(hyp_wp), score)
            if (i + 1) % 50 == 0:
                print(f"    [{tag}] {i+1} kalimat ...", flush=True)
    return times, hyps

def write_csv(path, meta, hyps):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "source", "hyp", "kalimat_koreksi", "gec_score"])
        miss = 0
        for m in meta:
            h = hyps.get(m["id"])
            if h is None:
                miss += 1
                w.writerow([m["id"], m["category"], m["src_orig"], m["src_orig"],
                            m["ref_orig"], ""])
            else:
                w.writerow([m["id"], m["category"], m["src_orig"], h[0],
                            m["ref_orig"], h[1]])
    print(f"  → {path} ({len(meta)} baris, {miss} fallback source)")

def bench_l2(csv_path, col="paddleocr_out"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from L2_normalizer import (load_validity_dict, augment_vocab_from_splits,
                               build_lookup, Normalizer, ALGO, DICTF, TRAIN, VAL)
    t0 = time.perf_counter()
    vocab = load_validity_dict(DICTF)
    augment_vocab_from_splits(vocab, [TRAIN, VAL])
    lookup = build_lookup(ALGO)
    nz = Normalizer(vocab, lookup)
    load_s = time.perf_counter() - t0
    texts = [r.get(col, "") or "" for r in csv.DictReader(open(csv_path, encoding="utf-8"))]
    times = []
    for tx in texts:
        t0 = time.perf_counter(); nz.norm_text(tx); times.append(time.perf_counter() - t0)
    return load_s, summarize(times), len(vocab), len(lookup)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--subset", default="test")
    ap.add_argument("--src-lang", default="src"); ap.add_argument("--tgt-lang", default="tgt")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0 = semua")
    ap.add_argument("--l2-csv", help="CSV berisi kolom paddleocr_out utk timing L2")
    ap.add_argument("--ext", default=os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/fairseq_extensions")
    ap.add_argument("--kg", default=os.environ.get("CASTLE_ROOT","/ssd-data1/sq2023/DidugaCASTLE_asli_n_paper6")+"/semantic_kg.json")
    ap.add_argument("--out-prefix", default="../results/deploy")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # paksa CPU
    from fairseq import checkpoint_utils, utils as fs_utils
    fs_utils.import_user_module(argparse.Namespace(user_dir=a.ext))

    rows = []  # utk CSV ringkasan
    print(f"== FASE D deployability benchmark (CPU, threads={a.threads}, beam={a.beam}) ==")

    # (0) L2 normalizer
    if a.l2_csv:
        load_s, tsum, nv, nl = bench_l2(a.l2_csv)
        print(f"\n[L2 normalizer] load {load_s:.2f}s | vocab {nv} + lookup {nl}")
        print(f"  latency/kalimat: median {tsum['median']*1000:.2f} ms | "
              f"p95 {tsum['p95']*1000:.2f} ms (n={tsum['n']})")
        rows.append(dict(component="L2_normalizer", precision="-",
                         size_mb="", median_ms=tsum['median']*1000,
                         p95_ms=tsum['p95']*1000, n=tsum['n']))

    # (1) load model FP32
    print(f"\n[load] {a.ckpt}")
    overrides = {"kg_path": a.kg, "cpu": True, "data": a.bin}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [a.ckpt], arg_overrides=overrides)
    for m in models: m.eval().cpu()
    model = models[0]
    n_params = sum(p.numel() for p in model.parameters())
    ckpt_mb = mb(os.path.getsize(a.ckpt))
    fp32_mb = state_dict_size_mb(model)
    print(f"  params {n_params/1e6:.1f}M | checkpoint {ckpt_mb:.1f} MB | "
          f"state_dict FP32 {fp32_mb:.1f} MB")
    task.load_dataset(a.subset)
    meta = [json.loads(l) for l in open(a.meta, encoding="utf-8") if l.strip()]

    # (2) FP32 latency + hyps
    print(f"\n[FP32] generate {a.subset} (batch=1) ...")
    t_fp, h_fp = run_model("fp32", models, task, a.subset, a.beam, a.limit)
    s = summarize(t_fp)
    print(f"  latency/kalimat: median {s['median']*1000:.0f} ms | p95 {s['p95']*1000:.0f} ms "
          f"| total {s['total']:.1f}s (n={s['n']})")
    write_csv(f"{a.out_prefix}_fp32.csv", meta, h_fp)
    rows.append(dict(component="CASTLE_corrector", precision="FP32",
                     size_mb=round(fp32_mb, 1), median_ms=s['median']*1000,
                     p95_ms=s['p95']*1000, n=s['n']))
    ram_fp32 = peak_ram_mb()

    # (3) INT8 dynamic quantization
    print("\n[INT8] dynamic quantization (nn.Linear → qint8) ...")
    # fairseq MultiheadAttention mengakses .bias proyeksi q/k/v/out secara
    # langsung (torch.cat) — Linear terkuantisasi mengekspos .bias sbg method →
    # kecualikan proyeksi attention; kuantisasi FFN + proyeksi output saja.
    import torch.quantization as tq
    skip = ("q_proj", "k_proj", "v_proj", "out_proj")
    qspec = {name: tq.default_dynamic_qconfig
             for name, mod in model.named_modules()
             if isinstance(mod, torch.nn.Linear) and not name.endswith(skip)}
    print(f"  quantize {len(qspec)} Linear (proyeksi attention dikecualikan)")
    qmodel = tq.quantize_dynamic(model, qconfig_spec=qspec)
    qmodel.eval()
    int8_mb = state_dict_size_mb(qmodel)
    print(f"  state_dict INT8 {int8_mb:.1f} MB ({int8_mb/fp32_mb:.0%} dari FP32)")
    print(f"[INT8] generate {a.subset} (batch=1) ...")
    t_q, h_q = run_model("int8", [qmodel], task, a.subset, a.beam, a.limit)
    sq = summarize(t_q)
    print(f"  latency/kalimat: median {sq['median']*1000:.0f} ms | p95 {sq['p95']*1000:.0f} ms "
          f"| total {sq['total']:.1f}s (n={sq['n']})")
    write_csv(f"{a.out_prefix}_int8.csv", meta, h_q)
    rows.append(dict(component="CASTLE_corrector", precision="INT8",
                     size_mb=round(int8_mb, 1), median_ms=sq['median']*1000,
                     p95_ms=sq['p95']*1000, n=sq['n']))

    # (4) ringkasan
    ram = peak_ram_mb()
    print(f"\n[RAM] peak RSS proses: {ram:.0f} MB (setelah FP32: {ram_fp32:.0f} MB)")
    outp = f"{a.out_prefix}_benchmark.csv"
    with open(outp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["component", "precision", "size_mb",
                                          "median_ms", "p95_ms", "n"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"→ {outp}")
    print("\nLangkah akhir: eval kedua CSV dgn eval_protocol_v2 (lihat docstring) "
          "utk degradasi CER FP32 vs INT8.")

if __name__ == "__main__":
    main()
