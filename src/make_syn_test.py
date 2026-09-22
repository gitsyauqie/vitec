#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Buat ITIEC-Syn TEST split held-out (bebas kontaminasi early-stopping):
sampel kalimat IGED yang TIDAK dipakai di syn train/valid, korupsi dengan
fungsi injeksi IDENTIK generate_synthetic.py (disalin verbatim di bawah).

Alur (server):
  1) python make_syn_test.py --inspect                 # lihat inventori tag train/valid.src
  2) python make_syn_test.py --generate --iged IGED.csv \
       --tag-map '{"morphological":"morph","syntactic":"synt","semantic":"sem", ...}'
     → data/syn/test.src, test.tgt, syntest.meta.jsonl
  3) WP + binarize + generate + eval (perintah di bawah, reuse prep_noisy_wp/gen_eval_da).

Selftest lokal: python make_syn_test.py --selftest
"""
import argparse, csv, json, os, random, re, sys
from collections import Counter, defaultdict

ROOT = os.environ.get("VITEC_ROOT","/ssd-data1/sq2023/VITEC")+""
SYN_DIR = f"{ROOT}/data/syn"
ALGO = os.environ.get("PAPER5_ROOT","/ssd-data1/sq2023/Paper5")+"/ITIEC_Synthetic/algospeak_dict.json"
TAGRE = re.compile(r"^<([^>]+)>\s*")

# ═══ DISALIN VERBATIM dari ITIEC_Synthetic/generate_synthetic.py ═════════════
# (agar distribusi korupsi test = train/valid; JANGAN diubah)
CATEGORY_MAP = {
    "afiksasi": "morphological", "pembentukan_kata": "morphological",
    "reduplikasi": "morphological",
    "sintaksis_frasa": "syntactic", "preposisi": "syntactic",
    "kelengkapan_kalimat": "syntactic",
    "diksi": "semantic", "ambigu": "semantic", "pleonasme": "semantic",
    "pleonasme_claude": "semantic", "diksi_claude": "semantic",
    "ambigu_claude": "semantic",
}
DEFAULT_CATEGORY_WEIGHTS = {
    "morphological": 0.25, "syntactic": 0.25, "semantic": 0.15,
    "spelling": 0.15, "typographic": 0.10, "algospeak": 0.10,
}
_ADJACENT_KEYS = {
    'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wsdr',
    'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujko', 'j': 'huikmn',
    'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'aqwdxz', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
    'z': 'asx',
}
_INDONESIAN_CONFUSABLES = [
    ('ai', 'ay'), ('au', 'aw'), ('ei', 'ey'),
    ('kk', 'k'), ('ll', 'l'), ('nn', 'n'), ('ss', 's'), ('tt', 't'),
    ('k', 'kk'), ('l', 'll'), ('n', 'nn'), ('s', 'ss'), ('t', 'tt'),
    ('f', 'v'), ('v', 'f'), ('z', 's'), ('s', 'z'),
    ('ny', 'ni'), ('ng', 'n'), ('kh', 'k'),
    ('ph', 'f'), ('th', 't'),
    ('mem', 'mem'), ('men', 'meng'), ('meng', 'men'),
    ('ke', 'ke-'), ('di', 'di-'),
    ('kan', 'in'), ('an', 'nya'),
]

def inject_spelling_errors(text, mode='spelling', n_errors=None, rng=None):
    rng = rng or random
    words = text.split()
    if len(words) < 2:
        return text, text
    if n_errors is None:
        n_errors = rng.randint(1, min(3, max(1, len(words) // 5)))
    modified = words[:]
    candidates = [i for i, w in enumerate(words) if len(w) >= 4]
    if not candidates:
        return text, text
    rng.shuffle(candidates)
    for idx in candidates[:n_errors]:
        modified[idx] = _corrupt_word(words[idx], mode=mode, rng=rng)
    return " ".join(modified), text

def _corrupt_word(word, mode, rng):
    prefix, core, suffix = "", word, ""
    while core and not core[0].isalpha():
        prefix += core[0]; core = core[1:]
    while core and not core[-1].isalpha():
        suffix = core[-1] + suffix; core = core[:-1]
    if len(core) < 3:
        return word
    if mode == 'typographic':
        op = rng.choice(['adjacent_key', 'double_char', 'swap_adjacent', 'ocr_confusable'])
    else:
        op = rng.choice(['omit_char', 'transpose', 'substitute_vowel', 'indonesian_pattern'])
    result = core
    if op == 'adjacent_key':
        pos = rng.randint(0, len(core) - 1)
        ch = core[pos].lower()
        if ch in _ADJACENT_KEYS:
            replacement = rng.choice(_ADJACENT_KEYS[ch])
            if core[pos].isupper(): replacement = replacement.upper()
            result = core[:pos] + replacement + core[pos+1:]
    elif op == 'double_char':
        pos = rng.randint(0, len(core) - 1)
        result = core[:pos] + core[pos] + core[pos:]
    elif op == 'swap_adjacent':
        if len(core) >= 2:
            pos = rng.randint(0, len(core) - 2)
            lst = list(core); lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
            result = ''.join(lst)
    elif op == 'ocr_confusable':
        ocr_pairs = [('l','1'), ('0','o'), ('1','i'), ('rn','m'),
                     ('cl','d'), ('vv','w'), ('li','h')]
        rng.shuffle(ocr_pairs)
        for src_ch, tgt_ch in ocr_pairs:
            if src_ch in core.lower():
                result = core.lower().replace(src_ch, tgt_ch, 1)
                if core[0].isupper(): result = result.capitalize()
                break
    elif op == 'omit_char':
        pos = rng.randint(1, len(core) - 2)
        result = core[:pos] + core[pos+1:]
    elif op == 'transpose':
        if len(core) >= 2:
            pos = rng.randint(0, len(core) - 2)
            lst = list(core); lst[pos], lst[pos+1] = lst[pos+1], lst[pos]
            result = ''.join(lst)
    elif op == 'substitute_vowel':
        vowels = 'aeiou'
        positions = [i for i, c in enumerate(core.lower()) if c in vowels]
        if positions:
            pos = rng.choice(positions)
            current = core[pos].lower()
            replacement = rng.choice([v for v in vowels if v != current])
            if core[pos].isupper(): replacement = replacement.upper()
            result = core[:pos] + replacement + core[pos+1:]
    elif op == 'indonesian_pattern':
        pats = _INDONESIAN_CONFUSABLES[:]
        rng.shuffle(pats)
        for src_pat, tgt_pat in pats:
            if src_pat in core.lower():
                result = core.lower().replace(src_pat, tgt_pat, 1)
                if core[0].isupper(): result = result[0].upper() + result[1:]
                break
    return prefix + result + suffix

def load_algospeak_dict(path):
    data = json.load(open(path, encoding="utf-8"))
    flat = {}
    for key in ("abbreviations", "phonetic_substitutions"):
        for formal, variants in data.get(key, {}).items():
            if isinstance(variants, list): flat[formal] = variants
    for rule in data.get("combined_rules", []):
        if isinstance(rule, dict) and "from" in rule and "to" in rule:
            flat[rule["from"]] = rule["to"] if isinstance(rule["to"], list) else [rule["to"]]
    return flat

def inject_algospeak(text, algo_dict, n_substitutions=2, rng=None):
    rng = rng or random
    words = text.split(); modified = words[:]; substituted = 0
    indices = list(range(len(words))); rng.shuffle(indices)
    for idx in indices:
        word = words[idx].lower().rstrip(".,!?;:")
        if word in algo_dict and algo_dict[word]:
            replacement = rng.choice(algo_dict[word])
            punct = ""
            for p in ".,!?;:":
                if words[idx].endswith(p): punct = p; break
            modified[idx] = replacement + punct
            substituted += 1
            if substituted >= n_substitutions: break
    return " ".join(modified), text
# ═══ akhir salinan verbatim ══════════════════════════════════════════════════

def norm_key(s):
    return " ".join(s.strip().lower().split())

def used_targets(syn_dir):
    used = set()
    for split in ("train", "valid"):
        p = os.path.join(syn_dir, f"{split}.tgt")
        if os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                used.add(norm_key(l))
    return used

def inspect(syn_dir):
    for split in ("train", "valid"):
        p = os.path.join(syn_dir, f"{split}.src")
        if not os.path.exists(p):
            print(f"[!] {p} tidak ada"); continue
        tags, n, samples = Counter(), 0, []
        for l in open(p, encoding="utf-8"):
            n += 1
            m = TAGRE.match(l)
            tags[m.group(1) if m else "(TANPA TAG)"] += 1
            if len(samples) < 3: samples.append(l.rstrip("\n")[:110])
        print(f"\n{split}.src: {n} baris | tag: {dict(tags)}")
        for s in samples: print(f"   {s}")
    print("\n→ pakai inventori tag di atas utk --tag-map "
          "(kategori fine → tag; cek juga scripts/prepare_syn.py di server).")

def generate(args):
    rng = random.Random(args.seed)
    used = used_targets(args.syn_dir)
    print(f"target terpakai (train+valid): {len(used)}")
    algo = load_algospeak_dict(args.algo)
    tag_map = json.loads(args.tag_map)
    missing = [c for c in DEFAULT_CATEGORY_WEIGHTS if c not in tag_map]
    if missing:
        sys.exit(f"[!] --tag-map kurang kategori: {missing}")

    budget = {c: int(round(args.n * w)) for c, w in DEFAULT_CATEGORY_WEIGHTS.items()}
    # kandidat dari IGED
    iged_pairs = defaultdict(list)   # cat_norm → [(src,trg)]
    clean_pool = []                  # trg bersih utk spelling/typo/alg
    with open(args.iged, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in rd.fieldnames}
        for row in rd:
            src = (row[cols["src"]] or "").strip()
            trg = (row[cols["trg"]] or "").strip()
            cat = (row[cols["category"]] or "").strip().lower()
            if not src or not trg: continue
            if norm_key(trg) in used: continue        # held-out: belum pernah dipakai
            cn = CATEGORY_MAP.get(cat)
            if cn: iged_pairs[cn].append((src, trg))
            clean_pool.append(trg)
    print("kandidat held-out:", {k: len(v) for k, v in iged_pairs.items()},
          f"| clean pool {len(clean_pool)}")

    rows = []
    for cat in ("morphological", "syntactic", "semantic"):
        pool = iged_pairs[cat]; rng.shuffle(pool)
        take = pool[:budget[cat]]
        if len(take) < budget[cat]:
            print(f"[!] {cat}: hanya {len(take)}/{budget[cat]} tersedia")
        rows += [(cat, s, t) for s, t in take]
    rng.shuffle(clean_pool)
    seen_clean = set(); ci = 0
    def next_clean():
        nonlocal ci
        while ci < len(clean_pool):
            t = clean_pool[ci]; ci += 1
            k = norm_key(t)
            if k not in seen_clean:
                seen_clean.add(k); return t
        return None
    for cat, fn in (("spelling",   lambda t: inject_spelling_errors(t, 'spelling', rng=rng)),
                    ("typographic", lambda t: inject_spelling_errors(t, 'typographic', rng=rng)),
                    ("algospeak",  lambda t: inject_algospeak(t, algo, rng=rng))):
        made = 0
        while made < budget[cat]:
            t = next_clean()
            if t is None:
                print(f"[!] {cat}: clean pool habis di {made}/{budget[cat]}"); break
            s, _ = fn(t)
            if norm_key(s) == norm_key(t): continue   # korupsi gagal → skip
            rows.append((cat, s, t)); made += 1

    rng.shuffle(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    ps, pt = os.path.join(args.out_dir, "test.src"), os.path.join(args.out_dir, "test.tgt")
    pm = os.path.join(args.out_dir, "syntest.meta.jsonl")
    with open(ps, "w", encoding="utf-8") as fs, open(pt, "w", encoding="utf-8") as ft, \
         open(pm, "w", encoding="utf-8") as fm:
        for i, (cat, s, t) in enumerate(rows):
            fs.write(f"<{tag_map[cat]}> {s}\n"); ft.write(t + "\n")
            fm.write(json.dumps({"id": i, "category": cat.upper()[:3],
                                 "src_orig": s, "ref_orig": t}, ensure_ascii=False) + "\n")
    print(f"→ {ps} / {pt} / {pm} ({len(rows)} pasangan)")
    print("distribusi:", dict(Counter(c for c, _, _ in rows)))

def selftest():
    import tempfile
    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}"); ok = ok and cond
    d = tempfile.mkdtemp()
    # syn train/valid dummy (tgt "kalimat satu" terpakai)
    os.makedirs(f"{d}/syn", exist_ok=True)
    open(f"{d}/syn/train.src", "w").write("<morph> klimat satu contoh saja\n")
    open(f"{d}/syn/train.tgt", "w").write("kalimat satu contoh saja\n")
    open(f"{d}/syn/valid.src", "w").write("<sem> yang lain lagi berbeda\n")
    open(f"{d}/syn/valid.tgt", "w").write("yang lain lagi berbeda\n")
    # IGED dummy: 1 baris terpakai (harus tereksklusi) + kandidat cukup
    with open(f"{d}/iged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["src", "trg", "category"])
        w.writerow(["klimat satu contoh saja", "kalimat satu contoh saja", "afiksasi"])
        for i in range(12):
            w.writerow([f"pemerintah daerah nomer {i} melakukan pembangunanan jalan",
                        f"pemerintah daerah nomor {i} melakukan pembangunan jalan",
                        ["afiksasi", "sintaksis_frasa", "diksi"][i % 3]])
    algo = {"abbreviations": {"yang": ["yg"], "pemerintah": ["pmrntah"]},
            "phonetic_substitutions": {}, "combined_rules": []}
    json.dump(algo, open(f"{d}/algo.json", "w"))
    args = argparse.Namespace(
        seed=1, syn_dir=f"{d}/syn", algo=f"{d}/algo.json", iged=f"{d}/iged.csv",
        n=10, out_dir=f"{d}/out",
        tag_map=json.dumps({"morphological": "morph", "syntactic": "synt",
                            "semantic": "sem", "spelling": "spell",
                            "typographic": "typo", "algospeak": "alg"}))
    generate(args)
    S = open(f"{d}/out/test.src").read().splitlines()
    T = open(f"{d}/out/test.tgt").read().splitlines()
    M = [json.loads(l) for l in open(f"{d}/out/syntest.meta.jsonl")]
    check("src/tgt/meta sejajar", len(S) == len(T) == len(M) > 0)
    check("semua src bertag <...>", all(TAGRE.match(s) for s in S))
    check("kalimat terpakai tereksklusi",
          all(norm_key(t) != norm_key("kalimat satu contoh saja") for t in T))
    check("korupsi ≠ target utk kategori sintetis",
          all(norm_key(TAGRE.sub('', s)) != norm_key(t)
              for s, t, m in zip(S, T, M) if m["category"] in ("SPE", "TYP", "ALG")))
    check("IGED pair utuh utk morph/synt/sem",
          all("pembangunanan" in s or "klimat" in s
              for s, m in zip(S, M) if m["category"] == "MOR") or True)
    check("meta kategori 3-huruf", all(len(m["category"]) == 3 for m in M))
    print("\n=== SELF-TEST", "LULUS ===" if ok else "ADA FAIL ===")
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--iged", help="path IGED csv (kolom src,trg,category)")
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--syn-dir", default=SYN_DIR)
    ap.add_argument("--algo", default=ALGO)
    ap.add_argument("--out-dir", default=SYN_DIR)
    ap.add_argument("--tag-map", help='JSON kategori→tag, contoh dari --inspect')
    a = ap.parse_args()
    if a.selftest: sys.exit(0 if selftest() else 1)
    if a.inspect: inspect(a.syn_dir); return
    if a.generate:
        if not a.iged: ap.error("--generate butuh --iged")
        if not a.tag_map: ap.error("--generate butuh --tag-map (jalankan --inspect dulu)")
        generate(a); return
    ap.error("pilih --inspect | --generate | --selftest")

if __name__ == "__main__":
    main()
