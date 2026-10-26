# Roadmap

**Goal:** a Spanish→English translator, built from scratch, that someone else can actually use, plus an honest linguist's account of what it gets right and where it fails.

**Now:** run 5 is the best model yet (26.6 BLEU / 49.5 chrF on the 1,800 clean test sentences); next come beam search, checkpoint averaging and a comparison with a production model.

## 1. A working training pipeline · Done
- Found and fixed the three original bugs: the end token was never trained, `<s>` was doubled, training diverged
- Config-driven training with live metrics in the Helm; Colab runs saved to Drive
- Multi-domain data (Tatoeba, TED, OpenSubtitles, Europarl) built by `build_mix.py`, with fixed per-register test sets
- Found and fixed an evaluation leak (test sentences' English in training) → `mix-v2`
- Pre-LayerNorm, spike rollback and gradient/fp16 logging after run 3 collapsed in epoch 9

**Done when:** a run trains to completion without collapsing and is scored on fixed, leak-free test sets. Reached with run 4 (24.0 BLEU / 47.3 chrF on 1,800 sentences).

## 2. A model worth using · In progress · run 5: 26.6 BLEU
- Run 5: run 4's setup for 20 epochs: +2.6 BLEU, every register better *(done)*
- Fix beam search: carry finished beams forward, add length normalisation; report greedy vs beam per register
- Checkpoint averaging (the mean of the last few epochs' weights), a cheap and usually reliable gain
- One decoding module (`src/decode.py`) used by `translate.py`, `eval.py`, `main.py` and the Helm adapter
- Reference point: score `Helsinki-NLP/opus-mt-es-en` on the same test sets, to state the gap honestly
- Optional, if Colab time allows: more data per register, back-translation, or a wider model

**Done when:** beam search scores at least as well as greedy on every register, and the README's results table shows the best run against runs 1–4 and opus-mt on the same sentences.

## 3. Usable by anyone · Next
- `translate.py` loads any run folder (architecture and token ids come from the checkpoint) and translates a sentence, a file or stdin
- A small Python API: `Translator("path").translate(["..."])`
- The model on the Hugging Face Hub with a model card: data, results per register, known failure modes, licence limits
- A Gradio demo on Hugging Face Spaces: translate, colour tokens by model confidence, show opus-mt's translation beside it
- Optional: int8/ONNX export with CPU latency numbers, for an offline phone or desktop app

**Done when:** a stranger can translate in the browser without installing anything, and `python translate.py "Llegué tarde porque se me olvidó el despertador."` works from a fresh clone with the published model.

## 4. The linguist's evaluation · Later
- A challenge set of ~150–300 hand-written Spanish sentences, each targeting one phenomenon: dropped subjects (pro-drop), *su/sus* → his/her/their/your, *se* constructions (*se me olvidó*, *se venden casas*), *ser/estar*, subjunctive, *gustar*-type verbs, clitic doubling, *tú/usted* register, idioms, false friends (*actualmente, embarazada, éxito*)
- Per-phenomenon accuracy for runs 2 onward and opus-mt
- Error analysis of ~100 test-set outputs with a small MQM-style scheme (accuracy, fluency, terminology, register), with examples
- `REPORT.md`: data, the bugs and what they taught, the divergence story, results, error analysis, limits

**Done when:** the README links a report that both a linguist and an ML engineer would find credible, and every claimed failure mode has an example and a count behind it.

## 5. Engineering hygiene · Later
- Tests (CPU, seconds): tokenizer special ids, teacher-forcing shapes, greedy stops at `</s>`, rollback logic, `build_mix.py` leak filter
- GitHub Actions running the tests on every push
- `requirements.txt` pinned; README commands verified from a fresh clone
- Git: tag the runs (`run-2` … `run-5`), descriptive commit messages from here on. No history rewrite needed: the repo is small and has never held data, checkpoints or secrets

**Done when:** a fresh clone installs, runs the tests green in CI, and reproduces a training run from the README alone.

## Decisions
- **Licences:** code under MIT; model weights under CC BY-NC 4.0, because TED2020 is CC BY-NC-ND and OpenSubtitles is research-use. The training data itself is never redistributed: `build_mix.py` rebuilds it from the original sources.
- **Test sets:** four fixed per-register sets of 500 sentences, the same for every model; Europarl's are run 1's original test sentences.
