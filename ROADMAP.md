# Roadmap

**Goal:** a Spanish-to-English translator, built from scratch, that someone else can actually use, plus an honest linguist's account of what it gets right and where it fails.

**Now:** with the fixed beam search, run 5 scores 27.9 BLEU / 50.5 chrF (opus-mt: 42.3 on the same sentences); milestone 2's bar is met, and checkpoint averaging is built and waits for a run 6 on Colab. The Gradio demo (`space/`) is built and runs locally; it isn't hosted, because Hugging Face now charges for Gradio Spaces (Nov 2026).

## 1. A working training pipeline · Done
- Found and fixed the three original bugs: the end token was never trained, `<s>` was doubled, training diverged
- Config-driven training with live metrics in the Helm; Colab runs saved to Drive
- Multi-domain data (Tatoeba, TED, OpenSubtitles, Europarl) built by `build_mix.py`, with fixed per-register test sets
- Found and fixed an evaluation leak (test sentences' English in training), fixed in `mix-v2`
- Pre-LayerNorm, spike rollback and gradient/fp16 logging after run 3 collapsed in epoch 9

**Done when:** a run trains to completion without collapsing and is scored on fixed, leak-free test sets. Reached with run 4 (24.0 BLEU / 47.3 chrF on 1,800 sentences).

## 2. A model worth using · In progress · run 5 and beam: 27.9 BLEU
- Run 5: run 4's setup for 20 epochs: +2.6 BLEU, every register better *(done)*
- Fixed beam search: finished hypotheses set aside, length normalisation (α 1.0, tuned on validation); +1.3 BLEU over greedy, better on all four registers *(done)*
- One decoding module (`src/decode.py`) used by `translate.py`, `eval.py`, `main.py` and the Helm adapter; the encoder now runs once per sentence (3.5 times faster greedy) *(done)*
- Reference point: `Helsinki-NLP/opus-mt-es-en` scores 42.3 / 62.6 on the same 1,800 sentences, an upper bound since it was trained on the same OPUS corpora *(done)*
- Checkpoint averaging: `average_last: N` saves the mean of the last N epochs and scores it beside the best checkpoint *(built; needs a run)*
- Run 6: run 5's config with `average_last: 5`, to measure averaging (one Colab session)
- Optional, if Colab time allows: more data per register, back-translation, or a wider model

**Done when:** beam search scores at least as well as greedy on every register, and the README's results table shows the best run against runs 1 to 4 and opus-mt on the same sentences.

## 3. Usable by anyone · In progress · demo built, waiting to upload
- `translate.py` loads any run folder (architecture and token ids come from the checkpoint) and translates a sentence, a file or stdin *(done for runs from October 2026 on)*
- A small Python API: `Translator("path").translate(["..."])`
- The model on the Hugging Face Hub with a model card: data, results per register, known failure modes, licence limits *(the Space's README is the card for now; the weights ship inside the Space)*
- A Gradio demo on Hugging Face Spaces: translate, colour tokens by model confidence, show opus-mt's translation beside it *(built Nov 2026 in `space/`: beam, greedy and opus-mt side by side, three random test sentences per register; tested in a replica of the Space's environment. Not hosted: Gradio Spaces need a paid plan since Nov 2026, so the options are paying, running the model in the browser (ONNX), or a free host. Confidence colouring later)*
- Optional: int8/ONNX export with CPU latency numbers, for an offline phone or desktop app

**Done when:** a stranger can translate in the browser without installing anything, and `python translate.py "Llegué tarde porque se me olvidó el despertador."` works from a fresh clone with the published model.

## 4. The linguist's evaluation · Later
- A challenge set of about 150 to 300 hand-written Spanish sentences, each targeting one phenomenon: dropped subjects (pro-drop), *su/sus* as his, her, their or your, *se* constructions (*se me olvidó*, *se venden casas*), *ser/estar*, subjunctive, *gustar*-type verbs, clitic doubling, *tú/usted* register, idioms, false friends (*actualmente, embarazada, éxito*)
- Per-phenomenon accuracy for runs 2 onward and opus-mt
- Error analysis of about 100 test-set outputs with a small MQM-style scheme (accuracy, fluency, terminology, register), with examples
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
