# Spanish-to-English Neural Machine Translation

An encoder-decoder Transformer for Spanish to English, built and trained from scratch in PyTorch. I
chose the pair because I speak both, so I can judge the output myself, not only by BLEU.

**27.9 BLEU / 50.5 chrF** with beam search on 1,800 held-out sentences from four registers.

## Results

| Register (test sentences) | Greedy | **Beam 5** | opus-mt-es-en |
|---|---|---|---|
| Parliament: Europarl (482) | 20.3 | **22.0** | 36.1 |
| Spoken: TED talks (488) | 25.7 | **26.7** | 42.3 |
| Dialogue: OpenSubtitles (443) | 26.7 | **28.2** | 42.6 |
| Everyday: Tatoeba (387) | 45.2 | **46.5** | 58.8 |
| **All: BLEU / chrF** | 26.6 / 49.5 | **27.9 / 50.5** | 42.3 / 62.6 |

The four test sets are fixed, 500 sentences each, minus 200 whose English had appeared in training.
opus-mt's 14-point lead is an upper bound: it was trained on the full OPUS collection these corpora
come from, so it has likely seen some test sentences. It isn't flawless either. For *Allison, sé mejor
que nadie…* it writes "Allison knows better than anyone", losing the first person that *sé* marks;
mine keeps it.

How the overall score got here, on the same 1,800 sentences (greedy):

| Run | Change | BLEU |
|---|---|---|
| 1 | Europarl only, end-of-sentence bug fixed | 6.7 |
| 2 | Europarl only, start-token and learning-rate fixes | 10.1 |
| 3 | multi-register mix of about 830k pairs | 23.8 |
| 4 | leak-free test sets, pre-LayerNorm, loss-spike rollback | 24.0 |
| 5 | 20 epochs | **26.6** |

## What it gets right, and wrong

```
ES: Llegué tarde porque se me olvidó el despertador.
EN: I was late because I forgot the alarm.
```
The "accidental" *se me olvidó* comes out as a plain "I forgot", which is what training on everyday
dialogue was for (the Europarl-only model produced nonsense here).

```
ES: Cuando era pequeña, mi abuela me contaba historias cada noche.
EN: When I was little, my grandmother told me stories every night.
```
Spanish drops the subject of *era*. An earlier run filled it with the nearest noun ("When my
grandmother was little…"); this one recovers the first person from context.

```
ES:   A pesar de los esfuerzos realizados durante los últimos años para mejorar la situación
      económica de las regiones más vulnerables, los resultados han sido desiguales y aún queda
      mucho trabajo por hacer.
Beam: Despite the efforts of the last few years to improve the economic situation in the regions,
      the results have been much more vulnerable, and they have still been very vulnerable.
```
Long sentences are the clear weakness: the first clause is faithful, then the model loses the
thread and repeats itself.

## How it's built

- **Model:** 6 encoder and 6 decoder layers, d_model 512, 8 heads, FFN 2048, pre-LayerNorm; 68.7M parameters. One
  SentencePiece BPE vocabulary (16k) shared by both languages.
- **Data:** about 830k pairs from four OPUS corpora, one per register: Tatoeba (everyday), TED2020
  (spoken), OpenSubtitles (dialogue), Europarl (parliament). Deduplicated across corpora, with test
  sentences excluded by both their Spanish and their English side (`build_mix.py`).
- **Training:** AdamW with a Noam warmup schedule, label smoothing 0.1, FP16 on a Colab T4, 20
  epochs (about 8.5 hours).
- **Decoding** (`src/decode.py`): batched greedy and beam search (width 5, length normalisation
  α = 1.0, both tuned on validation data). The encoder runs once per sentence.

The first working version scored about 1 BLEU because of bugs I found when retraining, each now
fixed and documented in the code:

- the end-of-sentence token was mapped to padding, so the model never learned to stop
- the decoder input started with two `<s>` tokens, so it never learned to start
- the learning-rate peak was too high for FP16 once there was enough data, and the loss went to NaN
- some test sentences leaked into training through Tatoeba's alternative translations
- post-LayerNorm training collapsed mid-run; pre-LayerNorm and an automatic rollback fixed it

## Running it

```bash
pip install -r requirements.txt
python build_mix.py --out data/data/mix-v2          # after downloading the OPUS corpora (URLs inside)
python main.py --config configs/mix-v2.yaml         # trains, then scores the four test sets
python translate.py --model <run>/model "Llegué tarde porque se me olvidó el despertador."
python eval.py --model <run>/model --clean           # the 1,800 sentences above, greedy and beam
```

Training needs a GPU (I use Colab); translating and scoring run on a CPU. `average_last: 5` in a
config also saves the average of the last five epochs' weights. `space/` builds a Gradio demo
(`python space/build.py --model <run>/model`, then `python space/build/app.py` to run it locally).

**Licence:** the code is MIT ([LICENSE](LICENSE)). The model was trained on TED2020 (CC BY-NC-ND)
and OpenSubtitles (research use), so the trained weights are for non-commercial use. Next steps, including a challenge set of Spanish phenomena (pro-drop, *se*
constructions, *ser/estar*, false friends), are in [ROADMAP.md](ROADMAP.md).
