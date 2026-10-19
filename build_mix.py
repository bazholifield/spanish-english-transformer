"""Build a multi-domain ES→EN training mix from OPUS corpora, plus fixed evaluation sets.

    python build_mix.py                      # writes data/data/mix-v1/

Sources (downloaded into data/data/opus/ beforehand; see SOURCES for the URLs):
  tatoeba    short everyday sentences, human-translated      CC BY 2.0 FR
  ted        TED talk transcripts, spoken register            CC BY-NC-ND 4.0
  subtitles  OpenSubtitles v2024: dialogue, slang (sampled)   research use, opensubtitles.org
  europarl   the original raw Europarl v7 in data/data/raw/   free, statmt.org

Output (data/data/mix-v1/):
  pools/<domain>.es|.en   cleaned, deduplicated, shuffled training candidates per domain;
                          a run takes the first max_pairs of each (configs/mix-v1.yaml)
  eval/<domain>.es|.en    500 held-out pairs per domain, fixed forever and never trained on.
                          Europarl's are run 1's test sentences, so every run stays comparable.
  manifest.json           sources, licenses, filters and counts
"""
import argparse
import json
import random
import re
import zipfile
from pathlib import Path

SOURCES = {
    "tatoeba": {"zip": "Tatoeba.zip", "files": ("Tatoeba.en-es.es", "Tatoeba.en-es.en"),
                "url": "https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/en-es.txt.zip",
                "license": "CC BY 2.0 FR"},
    "ted": {"zip": "TED2020.zip", "files": ("TED2020.en-es.es", "TED2020.en-es.en"),
            "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-es.txt.zip",
            "license": "CC BY-NC-ND 4.0"},
    "subtitles": {"zip": "OpenSubtitles.zip", "files": ("OpenSubtitles.en-es.es", "OpenSubtitles.en-es.en"),
                  "url": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-es.txt.zip",
                  "license": "research use (opensubtitles.org; see OPUS)"},
    "europarl": {"raw": ("europarl-v7.es-en.es", "europarl-v7.es-en.en"),
                 "url": "https://www.statmt.org/europarl/", "license": "free (statmt.org)"},
}
# Earlier pools win when the same Spanish sentence appears in several corpora.
PRIORITY = ["tatoeba", "ted", "europarl", "subtitles"]

TAGS = re.compile(r"</?[a-zA-Z][^>]*>|\{\\[^}]*\}")


def clean(s: str) -> str:
    s = TAGS.sub("", s)                 # subtitle formatting: <i>, {\an8}
    s = " ".join(s.split())
    return s.lstrip("-–— ").strip()     # dialogue dashes at the start of a subtitle line


def keep(es: str, en: str, max_chars: int) -> bool:
    if not es or not en or es == en:                     # empty, or left untranslated
        return False
    if len(es) > max_chars or len(en) > max_chars:
        return False
    ratio = len(es) / len(en)
    if not 0.5 <= ratio <= 2.0 and abs(len(es) - len(en)) > 12:  # misaligned pair
        return False
    if sum(c.isalpha() for c in es) < 2 or sum(c.isalpha() for c in en) < 2:
        return False
    return "♪" not in es and "♪" not in en              # song lyrics in subtitles


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def read_pairs(name: str, opus_dir: Path, raw_dir: Path, limit_lines: int | None):
    src = SOURCES[name]
    if "raw" in src:
        handles = [open(raw_dir / f, encoding="utf-8", errors="replace") for f in src["raw"]]
    else:
        z = zipfile.ZipFile(opus_dir / src["zip"])
        handles = [(line.decode("utf-8", "replace") for line in z.open(f)) for f in src["files"]]
    for i, (es, en) in enumerate(zip(*handles)):
        if limit_lines and i >= limit_lines:
            break
        yield clean(es), clean(en)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/data/mix-v1")
    ap.add_argument("--opus-dir", default="data/data/opus")
    ap.add_argument("--raw-dir", default="data/data/raw")
    ap.add_argument("--europarl-eval", help="test_es.txt from run 1 (its test_en.txt beside it)")
    ap.add_argument("--europarl-lines", type=int, default=600000, help="raw Europarl lines to read")
    ap.add_argument("--subtitles-sample", type=int, default=600000, help="pool size sampled from ~105M pairs")
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--max-chars", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    out, rng = Path(args.out), random.Random(args.seed)
    (out / "pools").mkdir(parents=True, exist_ok=True)
    (out / "eval").mkdir(parents=True, exist_ok=True)

    pools: dict[str, list[tuple[str, str]]] = {}
    stats: dict[str, dict] = {}
    for name in PRIORITY:
        seen, kept, read, eligible = set(), [], 0, 0
        sample = name == "subtitles"
        limit = args.europarl_lines if name == "europarl" else None
        for es, en in read_pairs(name, Path(args.opus_dir), Path(args.raw_dir), limit):
            read += 1
            if not keep(es, en, args.max_chars):
                continue
            eligible += 1
            if sample:
                # Reservoir sampling: a uniform sample of ~105M pairs in one pass and bounded memory.
                # (Remembering every pair to dedupe would need tens of GB; the sample is deduped below.)
                if len(kept) < args.subtitles_sample:
                    kept.append((es, en))
                else:
                    j = rng.randrange(eligible)
                    if j < args.subtitles_sample:
                        kept[j] = (es, en)
            elif (es, en) not in seen:
                seen.add((es, en))
                kept.append((es, en))
            if read % 5_000_000 == 0:
                print(f"  {name}: {read:,} read, {len(kept):,} kept", flush=True)
        if sample:
            kept = list(dict.fromkeys(kept))  # exact-pair dedupe of the sample, order kept
        rng.shuffle(kept)
        pools[name] = kept
        stats[name] = {"read": read, "clean": eligible, "unique_kept": len(kept)}
        print(f"{name}: {read:,} pairs read → {eligible:,} pass the filters → {len(kept):,} kept", flush=True)

    # Fixed evaluation sets: 500 per domain, removed from every pool (any domain) before training.
    evals: dict[str, list[tuple[str, str]]] = {}
    for name in PRIORITY:
        if name == "europarl" and args.europarl_eval:
            es_lines = Path(args.europarl_eval).read_text(encoding="utf-8").splitlines()
            en_lines = Path(args.europarl_eval).with_name("test_en.txt").read_text(encoding="utf-8").splitlines()
            held = list(zip(es_lines, en_lines))
            evals[name] = held[:args.eval_size]
            exclude = {norm(es) for es, _ in held}  # all of run 1's test set stays out of training
            pools[name] = [p for p in pools[name] if norm(p[0]) not in exclude]
        else:
            evals[name], pools[name] = pools[name][:args.eval_size], pools[name][args.eval_size:]
    # Exclude eval sentences by BOTH sides: Tatoeba often has several Spanish versions of one English
    # sentence, so a Spanish-only check let ~22% of its eval targets into training (found in run 3).
    held_out = {norm(es) for pairs in evals.values() for es, _ in pairs}
    held_out_en = {norm(en) for pairs in evals.values() for _, en in pairs}

    taken: set[str] = set()
    for name in PRIORITY:  # cross-corpus dedupe by Spanish sentence, higher-priority pools first
        before = len(pools[name])
        pool = []
        for es, en in pools[name]:
            k = norm(es)
            if k in held_out or k in taken or norm(en) in held_out_en:
                continue
            taken.add(k)
            pool.append((es, en))
        pools[name] = pool
        stats[name].update(pool=len(pool), eval=len(evals[name]), dropped_as_duplicates=before - len(pool))
        for split, pairs in (("pools", pool), ("eval", evals[name])):
            with open(out / split / f"{name}.es", "w", encoding="utf-8") as f_es, \
                 open(out / split / f"{name}.en", "w", encoding="utf-8") as f_en:
                for es, en in pairs:
                    f_es.write(es + "\n")
                    f_en.write(en + "\n")
        print(f"{name}: pool {len(pool):,}, eval {len(evals[name])}", flush=True)

    manifest = {
        "built_by": "build_mix.py", "seed": args.seed, "max_chars": args.max_chars, "eval_size": args.eval_size,
        "europarl_lines": args.europarl_lines, "subtitles_sample": args.subtitles_sample,
        "europarl_eval": "run 1's test set" if args.europarl_eval else "random",
        "filters": ["strip subtitle tags and leading dialogue dashes", "non-empty, not identical",
                    f"both sides ≤ {args.max_chars} chars", "length ratio 0.5–2 (or within 12 chars)",
                    "≥ 2 letters per side", "no ♪ lyrics", "exact-pair dedupe within a corpus",
                    "Spanish-side dedupe across corpora", "eval sentences excluded everywhere, by Spanish and English side"],
        "sources": {n: {"url": SOURCES[n]["url"], "license": SOURCES[n]["license"], **stats[n]} for n in PRIORITY},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}/pools, {out}/eval and manifest.json")


if __name__ == "__main__":
    main()
