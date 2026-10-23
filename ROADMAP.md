# Roadmap

**Goal:** a Spanish→English translator, built from scratch, that someone else can actually use,
plus an honest, linguist's account of what it gets right and where it fails.

Each milestone ends with exit criteria ("someone can do X"). Quality comes first, then making it
usable, then the analysis that makes it a portfolio piece rather than a tutorial.

## Done so far (2026)

- [x] Found and fixed the three original bugs: end token never trained, doubled `<s>`, divergence (README "What I fixed")
- [x] Config-driven training with live metrics in the Helm; Colab runs saved to Drive
- [x] Multi-domain training data (Tatoeba, TED, OpenSubtitles, Europarl) with `build_mix.py`
- [x] Fixed per-domain evaluation sets, the same for every model
- [x] Found and fixed an evaluation leak (English side of the eval sets in training) → `mix-v2`
- [x] Pre-LayerNorm, spike rollback and gradient/fp16 logging after run 3 collapsed in epoch 9

## Milestone 1: a model worth using

- [ ] Run 4 (pre-LN, rollback, `mix-v2`) beats run 3's epoch-8 checkpoint on every clean eval set
- [ ] Fix beam search: carry finished beams forward, length normalisation; report greedy vs beam per domain
- [ ] Checkpoint averaging (mean of the last few epochs' weights), a cheap and usually reliable gain
- [ ] One decoding module (`src/decode.py`) used by `translate.py`, `eval.py`, `main.py` and the Helm adapter
- [ ] Reference point: score `Helsinki-NLP/opus-mt-es-en` (a production-grade model) on the same eval sets, to state the gap honestly
- [ ] Optional, if Colab time allows: more data per domain, back-translation, or a wider model

**Exit criteria**
- Beam search scores at least as well as greedy on every domain.
- The README's results table shows run 4 against runs 1–3 and opus-mt on the same sentences.

## Milestone 2: usable by anyone

- [ ] `translate.py` loads any run folder (architecture and token ids come from the checkpoint), translates a sentence, a file or stdin
- [ ] A small Python API: `Translator("path").translate(["..."])`
- [ ] Model on the Hugging Face Hub with a model card: data, results per domain, known failure modes, licence limits (see below)
- [ ] Gradio demo on Hugging Face Spaces (free CPU): translate, colour tokens by model confidence, show opus-mt's translation beside it
- [ ] Optional: int8/ONNX export with CPU latency numbers, for an offline phone or desktop app

**Exit criteria**
- A stranger can translate in the browser without installing anything.
- `python translate.py "Llegué tarde porque se me olvidó el despertador."` works from a fresh clone with the published model.

**Licences (decided):** code under MIT; model weights under CC BY-NC 4.0, because TED2020 is
CC BY-NC-ND and OpenSubtitles is research-use, with the reason on the model card. The training data
itself is never redistributed: `build_mix.py` rebuilds it from the original sources.

## Milestone 3: the linguist's evaluation

- [ ] A challenge set of ~150–300 hand-written Spanish sentences, each targeting one phenomenon:
  dropped subjects (pro-drop), *su/sus* → his/her/their/your, *se* constructions (*se me olvidó*,
  *se venden casas*), *ser/estar*, subjunctive, *gustar*-type verbs, clitic doubling, *tú/usted*
  register, idioms, false friends (*actualmente, embarazada, éxito*)
- [ ] Per-phenomenon accuracy for runs 2–4 and opus-mt (acceptable-translation lists or pattern checks per item)
- [ ] Error analysis of ~100 eval-set outputs with a small MQM-style scheme (accuracy, fluency, terminology, register), with examples
- [ ] `REPORT.md`: data, the bugs and what they taught, the divergence story, results, error analysis, limits

**Exit criteria**
- The README links a report that both a linguist and an ML engineer would find credible.
- Every claim in the report about a failure mode has an example and a count behind it.

## Milestone 4: engineering hygiene

- [ ] Tests (CPU, seconds): tokenizer special ids, teacher-forcing shapes, greedy stops at `</s>`, rollback logic, `build_mix.py` leak filter
- [ ] GitHub Actions running the tests on every push
- [ ] `requirements.txt` pinned; README commands verified from a fresh clone
- [ ] LICENSE (MIT) for the code; data and model licence notes in the README and model card
- [ ] Git: tag the runs (`run-2`, `run-3`, `run-4`), descriptive commit messages from here on. No history rewrite needed: the repo is 380 KB and has never held data, checkpoints or secrets.
