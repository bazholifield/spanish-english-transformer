"""Assemble the Hugging Face Space folder from a trained run.

    python space/build.py --model <run folder>       # writes space/build/ (gitignored)
    python space/build/app.py                         # try it locally

The checkpoint main.py saves is ~835 MB because it carries the optimizer state for resuming
training. The Space only needs the weights, so they're saved in half precision (~140 MB);
load_model copies them into the full-precision model unchanged, so translations are the same up to
rounding. Everything else is copied as is: the model code becomes the `mt` package next to app.py.
"""

import argparse
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).parent
SRC = HERE.parent / "src"
CODE = ["__init__.py", "architecture.py", "utils.py", "tokenizer.py", "decode.py"]
SPACE_FILES = ["app.py", "examples.json", "requirements.txt", "README.md"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run folder with best_transformer.pt and spm_shared.model")
    ap.add_argument("--out", default=str(HERE / "build"))
    args = ap.parse_args()
    run, out = Path(args.model), Path(args.out)

    (out / "mt").mkdir(parents=True, exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    for name in CODE:
        shutil.copy2(SRC / name, out / "mt" / name)
    for name in SPACE_FILES:
        shutil.copy2(HERE / name, out / name)
    shutil.copy2(run / "spm_shared.model", out / "model" / "spm_shared.model")

    ckpt = torch.load(run / "best_transformer.pt", map_location="cpu")
    slim = {k: ckpt[k] for k in ("arch", "special_ids", "decoder_input", "epoch", "val_loss")}
    slim["model_state"] = {k: v.half() if v.is_floating_point() else v for k, v in ckpt["model_state"].items()}
    torch.save(slim, out / "model" / "best_transformer.pt")

    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder ready: {out} ({size:.0f} MB, epoch {slim['epoch']}, val loss {slim['val_loss']:.3f})")


if __name__ == "__main__":
    main()
