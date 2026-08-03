"""Score a TTS model by asking the phoneme ASR to listen back to it.

The model we used to build the targets can also judge what a TTS trained on them produces:
synthesise held-out phoneme sequences, run the ASR over the synthesised audio, and measure the
unit error rate between what we asked for and what comes back. It is a direct, automatic test of
whether the TTS is actually articulating the phonemes rather than emitting fluent mush, and it
needs no human listening and no reference recording.

**It must be read against a baseline, not against zero.** The ASR has ~17% UER on genuine Asante
Twi, so a perfect vocoder replaying real speech would still score ~17%. `--real-audio` computes
that floor on the same held-out clips, and the number that matters is the gap:

    round-trip UER on synthesised audio  -  UER on the real recordings

At a gap near zero the TTS is as intelligible to the ASR as a real speaker. A large gap means
phonemes are being dropped or slurred. Tracking it across checkpoints gives a convergence signal
that correlates with intelligibility, which validation loss does not.

Caveats worth keeping in mind when reading the output:

  * It rewards clarity, not naturalness. A robotic but crisp voice can beat a warm mumbling one.
    Use it to catch regressions and pick checkpoints, not to declare a voice good.
  * Both sides share a model, so systematic ASR biases cancel in the gap but not in the raw UER.
  * Punctuation is excluded by default: it has no acoustic realisation and the ASR guesses it at
    ~30% error, which would swamp the phoneme signal.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def edit_distance(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def is_punct(u: str) -> bool:
    return len(u) == 1 and unicodedata.category(u).startswith("P")


def strip_punct(seq: list[str]) -> list[str]:
    return [u for u in seq if not is_punct(u)]


def recognise(wavs: list[tuple[np.ndarray, int]], model_root: Path, device: str,
              budget: int, max_rows: int) -> list[list[str]]:
    """Batch the ASR over a list of (waveform, sample_rate)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from phonemise import MODEL_SR, ctc_collapse, load_model, make_batches, run_batch

    model, keep, vocab = load_model(model_root, device)
    lens16 = [round(len(w) * MODEL_SR / sr) for w, sr in wavs]
    out: list[list[str]] = [None] * len(wavs)  # type: ignore[list-item]
    cache: dict = {}

    by_sr: dict[int, list[int]] = {}
    for i, (_, sr) in enumerate(wavs):
        by_sr.setdefault(sr, []).append(i)

    for sr, idx in by_sr.items():
        for b in make_batches(lens16, idx, budget, max_rows):
            res = run_batch(model, keep, [wavs[i][0] for i in b], [sr] * len(b), cache,
                            device)
            for i, ids in zip(b, res):
                out[i] = ctc_collapse(ids, vocab)
    return out


def load_wavs(paths: list[Path]) -> list[tuple[np.ndarray, int]]:
    got = []
    for p in paths:
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        got.append((w, int(sr)))
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="TSV with columns id and ipa: what the TTS was asked to say")
    ap.add_argument("--synth-dir", required=True,
                    help="directory of <id>.wav produced by the TTS")
    ap.add_argument("--real-dir", default=None,
                    help="directory of the real <id>.wav, to measure the ASR's own floor")
    ap.add_argument("--model", default="ghananlpcommunity/ghana-speech-phoneme-asr")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--keep-punct", action="store_true")
    ap.add_argument("--group-col", default="language",
                    help="manifest column to break results down by, if present")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    ap.add_argument("--budget", type=int, default=16000 * 900)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True

    import csv
    with open(args.manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                   escapechar="\\"))

    synth = Path(args.synth_dir)
    rows = [r for r in rows if (synth / f"{r['id']}.wav").exists()][: args.limit]
    if not rows:
        raise SystemExit(f"no synthesised wavs found in {synth}")
    print(f"scoring {len(rows)} utterances")

    from phonemise import resolve_model
    root = resolve_model(args.model)

    want = [r["ipa"].split(" ") for r in rows]
    if not args.keep_punct:
        want = [strip_punct(s) for s in want]

    hyp_s = recognise(load_wavs([synth / f"{r['id']}.wav" for r in rows]), root,
                      args.device, args.budget, args.max_rows)
    if not args.keep_punct:
        hyp_s = [strip_punct(h) for h in hyp_s]

    def uer_of(refs, hyps, idx=None):
        idx = range(len(refs)) if idx is None else idx
        e = sum(edit_distance(refs[i], hyps[i]) for i in idx)
        n = sum(len(refs[i]) for i in idx)
        return e / max(n, 1)

    result = {"n": len(rows), "synth_uer": round(uer_of(want, hyp_s), 4)}

    if args.real_dir:
        real = Path(args.real_dir)
        avail = [i for i, r in enumerate(rows) if (real / f"{r['id']}.wav").exists()]
        if avail:
            hyp_r = recognise(load_wavs([real / f"{rows[i]['id']}.wav" for i in avail]),
                              root, args.device, args.budget, args.max_rows)
            if not args.keep_punct:
                hyp_r = [strip_punct(h) for h in hyp_r]
            ref_sub = [want[i] for i in avail]
            result["real_uer"] = round(uer_of(ref_sub, hyp_r), 4)
            result["gap"] = round(result["synth_uer"] - result["real_uer"], 4)
            result["n_real"] = len(avail)

    groups = {}
    if args.group_col and args.group_col in rows[0]:
        for g in sorted({r[args.group_col] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[args.group_col] == g]
            groups[g] = {"n": len(idx), "synth_uer": round(uer_of(want, hyp_s, idx), 4)}
        result["by_" + args.group_col] = groups

    print(f"\nround-trip UER on synthesised audio : {result['synth_uer']:.2%}")
    if "real_uer" in result:
        print(f"UER on the real recordings (floor)  : {result['real_uer']:.2%}"
              f"   [n={result['n_real']}]")
        print(f"gap                                : {result['gap']:+.2%}"
              f"   <- the number to watch")
    for g, v in groups.items():
        print(f"  {g:12} n={v['n']:4}  synth UER {v['synth_uer']:.2%}")

    empty = sum(1 for h in hyp_s if not h)
    if empty:
        print(f"\nwarning: the ASR heard nothing in {empty} synthesised clips")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
