"""Sanity-check the batched GPU path before spending a GPU-hour on a corpus.

Three failure modes here produce plausible IPA rather than a crash, so each gets a number:

  batch vs single   w2v2's conv feature extractor and conv positional encoding are not
                    masked, so padding bleeds across clips in a batch. Length-sorted
                    batching keeps this tiny; this measures how tiny.
  sherpa vs torch   the int8 ONNX model through sherpa-onnx is the reference decoder from
                    the model card. Both are fed through the same torchaudio resampler, so
                    the only remaining difference is quantisation. A large gap here means
                    the front-end is wrong — missing waveform normalisation shows up as
                    ~12% UER.
  regression        optional: compare against an existing results parquet, to confirm a
                    refactor changed nothing.

Run this on a new dataset before the full sweep. It is cheap.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from phonemise import (MODEL_SR, ctc_collapse, load_model, make_batches, read_tokens,
                       resolve_model, run_batch)


def edit_distance(a, b):
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def uer(refs, hyps):
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / max(
        sum(len(r) for r in refs), 1)


def report(name, refs, hyps, ids=None, show=0):
    ex = sum(a == b for a, b in zip(refs, hyps))
    print(f"{name:38} {ex}/{len(refs)} exact, UER {uer(refs, hyps):.4%}")
    shown = 0
    for i, (a, b) in enumerate(zip(refs, hyps)):
        if a != b and shown < show:
            print(f"    [{i}] {ids[i] if ids else i}")
            print(f"        ref {' '.join(a)}")
            print(f"        hyp {' '.join(b)}")
            shown += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ghananlpcommunity/ghana-speech-phoneme-asr")
    ap.add_argument("--shard", required=True, help="a parquet shard with embedded audio")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-sherpa", action="store_true")
    ap.add_argument("--against", default=None,
                    help="results parquet from a previous run, to check for regressions")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    root = resolve_model(args.model)
    model, keep, vocab = load_model(root, args.device)

    cols = [args.audio_col] + ([args.text_col] if args.text_col else [])
    t = pq.read_table(args.shard, columns=cols).slice(0, args.n).to_pydict()
    waves, srs, ids = [], [], []
    for a in t[args.audio_col]:
        w, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        waves.append(w)
        srs.append(int(sr))
        ids.append(a["path"])
    texts = t[args.text_col] if args.text_col else [None] * len(waves)
    sr = srs[0]

    cache: dict = {}
    single = [ctc_collapse(run_batch(model, keep, [w], [sr], cache, args.device)[0], vocab)
              for w in waves]

    lens16 = [round(len(w) * MODEL_SR / sr) for w in waves]
    print()

    # Realistic batching, and deliberately tight so several batches actually form.
    for label, budget, rows in [("production batching", MODEL_SR * 900, 256),
                                ("tight batching (8 rows)", MODEL_SR * 900, 8)]:
        out: list = [None] * len(waves)
        for b in make_batches(lens16, list(range(len(waves))), budget, rows):
            for i, o in zip(b, run_batch(model, keep, [waves[i] for i in b],
                                         [sr] * len(b), cache, args.device)):
                out[i] = ctc_collapse(o, vocab)
        report(f"{label} vs single", single, out, ids, show=2)
        if label.startswith("production"):
            prod = out

    # One batch containing everything, short clips next to long: worst-case padding.
    allb = [ctc_collapse(o, vocab)
            for o in run_batch(model, keep, waves, [sr] * len(waves), cache, args.device)]
    report("one unsorted batch vs single", single, allb)

    if not args.skip_sherpa:
        import sherpa_onnx
        import torchaudio
        onnx = root / "onnx/model.int8.onnx"
        if not onnx.exists():
            from huggingface_hub import hf_hub_download
            onnx = Path(hf_hub_download(args.model, "onnx/model.int8.onnx"))
        rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=str(onnx), tokens=str(root / "onnx/tokens.txt"), num_threads=8)
        ref = []
        for w in waves:
            w16 = (w if sr == MODEL_SR else torchaudio.functional.resample(
                torch.from_numpy(w), sr, MODEL_SR).numpy()).astype("float32")
            s = rec.create_stream()
            s.accept_waveform(MODEL_SR, w16)
            rec.decode_stream(s)
            ref.append(list(s.result.tokens))
        report("sherpa int8 (reference) vs torch", ref, single)

    if args.against:
        prev = pq.read_table(args.against, columns=["id", "ipa_units"]).to_pydict()
        want = dict(zip(prev["id"], prev["ipa_units"]))
        pairs = [(want[Path(i).stem], p) for i, p in zip(ids, prod)
                 if Path(i).stem in want]
        if pairs:
            report(f"regression vs {Path(args.against).name}",
                   [a for a, _ in pairs], [b for _, b in pairs], show=2)
        else:
            print("regression: no overlapping ids")

    print("\nsample output")
    for i in range(min(5, len(waves))):
        print(f"  {ids[i]}  ({len(waves[i])/sr:.1f}s)")
        print(f"    text {texts[i]}")
        print(f"    ipa  {' '.join(prod[i])}")


if __name__ == "__main__":
    main()
