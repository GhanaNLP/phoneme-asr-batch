"""Graft LibriTTS-R's phoneme embeddings for newly added symbols into a running checkpoint.

Switching English from a Ghanaian-folded inventory to canonical espeak IPA introduces symbols
the run has never seen (θ, æ, ɹ, ɜ, ɑ, ʌ, ɚ, the diphthongs, stress and length marks). Left
random, each has to be learned from scratch.

They do not have to be. The pretrained LibriTTS-R checkpoint spent 1.9M steps learning exactly
these phonemes, so its embedding row for θ is a good starting vector for our θ. The id differs —
some of Piper's slots were already taken by Twi units — but an embedding is a vector at an index,
so the row is copied to whatever index we assigned.

Everything else is left alone: Twi ids keep their meaning, the speaker table keeps all 1,555
voices, and the decoder keeps the audio quality already learned. That is the point of doing this
as surgery rather than a restart.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

EMB = "model_g.enc_p.emb.weight"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint to continue from")
    ap.add_argument("--pretrained", required=True, help="LibriTTS-R checkpoint")
    ap.add_argument("--graft", required=True, help="graft.json from retarget_g2p.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale the copied rows to match the target's own row magnitude")
    args = ap.parse_args()

    graft = json.loads(Path(args.graft).read_text())
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    pre = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    psd = pre.get("state_dict", pre)

    if EMB not in sd or EMB not in psd:
        raise SystemExit(f"{EMB} missing from one of the checkpoints")
    cur, src = sd[EMB], psd[EMB]
    print(f"target emb {tuple(cur.shape)}, source emb {tuple(src.shape)}")

    # The two runs have different embedding scales, so copying raw vectors can land far off
    # the distribution the encoder currently expects. Rescale to the target's typical norm.
    cur_norm = cur.norm(dim=1).median().item()
    src_norm = src.norm(dim=1).median().item()
    k = args.scale * (cur_norm / max(src_norm, 1e-6))
    print(f"median row norm: target {cur_norm:.4f}, source {src_norm:.4f} -> scaling by {k:.4f}")

    done, skipped = [], []
    for sym, m in graft.items():
        n, p = int(m["new_id"]), int(m["piper_id"])
        if n >= cur.shape[0] or p >= src.shape[0]:
            skipped.append(sym)
            continue
        cur[n] = src[p] * k
        done.append(f"{sym}->{n}")

    sd[EMB] = cur
    print(f"grafted {len(done)}: {', '.join(done)}")
    if skipped:
        print(f"skipped (out of range): {skipped}")

    # Optimiser moments for the embedding now refer to rows that changed meaning. Dropping
    # them costs a short warm-up and avoids stale momentum fighting the new vectors.
    ckpt.pop("optimizer_states", None)
    ckpt.pop("lr_schedulers", None)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
