"""Resize a pretrained Piper checkpoint's speaker table to our speaker count.

Everything in the LibriTTS-R checkpoint transfers except `emb_g`, the speaker embedding, whose
first dimension is its own 904 speakers. Shapes must match to load, so it gets rebuilt.

New rows are seeded by *sampling actual pretrained speaker vectors* rather than random noise.
Each of our pseudo-speakers therefore starts from a real, already-coherent voice and moves
toward its target, instead of starting from a point the decoder has never seen and having to
find voice space from scratch. Sampling is done without replacement while rows last, so we do
not start several speakers from an identical vector.

The phoneme embedding is deliberately left alone: it is (256, 192), our ids top out at 175, and
59 of our units were mapped onto the ids Piper already uses for the same sound, so those rows
carry real phonetic knowledge into the finetune.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

EMB_G = "model_g.emb_g.weight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--manifest", default=None,
                    help="count distinct speakers from this TSV instead of --num-speakers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n = args.num_speakers
    if n is None:
        if not args.manifest:
            raise SystemExit("pass --num-speakers or --manifest")
        with open(args.manifest, encoding="utf-8") as fh:
            spk = {r["speaker"] for r in csv.DictReader(fh, delimiter="\t",
                                                        quoting=csv.QUOTE_NONE,
                                                        escapechar="\\")}
        n = len(spk)
        print(f"{n} distinct speakers in {args.manifest}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if EMB_G not in sd:
        raise SystemExit(f"{EMB_G} not in checkpoint; is this a multi-speaker model?")

    old = sd[EMB_G]
    n_old, dim = old.shape
    print(f"emb_g: {n_old} -> {n} speakers (dim {dim})")

    g = torch.Generator().manual_seed(args.seed)
    if n <= n_old:
        pick = torch.randperm(n_old, generator=g)[:n]
    else:
        # Use every pretrained voice once, then top up with repeats plus a little jitter so
        # the duplicated speakers do not start life perfectly tied together.
        extra = torch.randint(0, n_old, (n - n_old,), generator=g)
        pick = torch.cat([torch.randperm(n_old, generator=g), extra])
    new = old[pick].clone()
    if n > n_old:
        jitter = 0.01 * old.std() * torch.randn((n - n_old, dim), generator=g)
        new[n_old:] += jitter

    sd[EMB_G] = new

    for key in ("num_speakers", "n_speakers"):
        hp = ckpt.get("hyper_parameters")
        if isinstance(hp, dict) and key in hp:
            hp[key] = n
            print(f"hyper_parameters[{key}] = {n}")

    # A resized speaker table makes the optimiser's moments for that tensor meaningless, and
    # a stale global step would confuse the LR schedule on resume.
    ckpt.pop("optimizer_states", None)
    ckpt.pop("lr_schedulers", None)
    ckpt["global_step"] = 0
    ckpt["epoch"] = 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
