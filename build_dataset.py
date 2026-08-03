"""Merge per-shard results into a sidecar dataset and optionally push it to the Hub.

Sidecar means: no audio. One row per source clip, keyed by the source clip's filename, so
users load the audio dataset and this one and join on `id`. Republishing 30 GB of waveforms
just to attach a phoneme string would be waste, and it would fork the audio into a second
copy that can drift from the original.

Splits are taken from the source shard filenames, so the sidecar's train/test boundary is
the same as the source dataset's by construction rather than by assumption.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CARD = """---
license: cc-by-nc-4.0
language:
- tw
- ak
task_categories:
- automatic-speech-recognition
- text-to-speech
tags:
- phonemes
- ipa
- twi
- akan
- g2p
- sidecar
size_categories:
- 100K<n<1M
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: test
    path: data/test-*.parquet
---

# new-twi-tts-aligned — IPA phoneme transcriptions

Machine-generated IPA phoneme transcriptions for every clip in
[`ghanaopendata/new-twi-tts-aligned`]({src_url}), produced with
[`ghananlpcommunity/ghana-speech-phoneme-asr`]({model_url}).

This is a **sidecar dataset: it contains no audio.** It is keyed by the source dataset's
clip filename so you can attach phonemes to the audio without a second 30 GB copy of the
waveforms.

## Contents

{stats_table}

## Columns

| column | type | meaning |
|---|---|---|
| `id` | string | clip id, e.g. `segment_042811` — the join key |
| `audio_path` | string | source `audio.path`, e.g. `segment_042811.wav` |
| `shard` | string | source parquet shard the clip came from |
| `row` | int32 | row index within that shard |
| `duration` | float32 | clip length in seconds |
| `text` | string | the source orthographic Twi text, carried through unchanged |
| `ipa` | string | phonemes, **space-separated** |
| `ipa_units` | list[string] | the same phonemes as a list |
| `n_units` | int32 | number of phoneme units |

## Joining it to the audio

```python
from datasets import load_dataset

audio = load_dataset("{src}", split="train")
ipa = load_dataset("{repo}", split="train")

lookup = dict(zip(ipa["id"], ipa["ipa"]))
audio = audio.map(lambda r: {{"ipa": lookup[r["audio"]["path"].removesuffix(".wav")]}})
```

## Read `ipa`, not the characters

Many units are multi-character — `kʰ`, `t͡ʃ`, `k͡p`, `hʷ`, `iː`. **Split `ipa` on spaces**;
iterating over the string by character will tear these apart and corrupt the inventory.

```python
units = row["ipa"].split(" ")     # correct
units = list(row["ipa"])          # wrong — breaks k͡p into three characters
```

## How it was made

Greedy CTC decoding of the fairseq2 checkpoint in bf16 on a single H200, in
length-sorted batches. Code, including the validation harness:
{code_url}

## Accuracy, and what to expect

These are **model predictions, not verified ground truth.** On its own held-out dev set the
model scores {uer_note}. Treat this dataset as a strong starting point for TTS phoneme
targets or pronunciation analysis, not as a gold lexicon.

Two known properties worth knowing before you train on it:

- **Punctuation is guessed.** The model emits punctuation marks, but they have no acoustic
  realisation, so its punctuation error rate is high (~30%) even where phonemes are good.
  Strip punctuation if you do not want it: `[u for u in units if u.isalpha() or len(u) > 1]`
  is not sufficient — filter against a known punctuation set.
- **The IPA reflects what was said, not the orthography.** Where a speaker elides or
  reduces, the phonemes follow the speech and will not match a grapheme-to-phoneme
  rendering of `text`. That is the point of using an acoustic model, but it means
  `ipa` and `text` will legitimately disagree.

Batching introduces a further ~0.2% unit error versus decoding each clip alone, because
wav2vec2's convolutional front-end is not padding-masked. This is far below the model's own
error rate.

## License

`cc-by-nc-4.0`, inherited from the source audio dataset.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="out/shards")
    ap.add_argument("--out", default="out/dataset")
    ap.add_argument("--src", default="ghanaopendata/new-twi-tts-aligned")
    ap.add_argument("--model", default="ghananlpcommunity/ghana-speech-phoneme-asr")
    ap.add_argument("--repo", default=None, help="HF dataset repo id to push to")
    ap.add_argument("--code-url", default="https://github.com/GhanaNLP/phoneme-asr-batch")
    ap.add_argument("--uer-note",
                    default="17.1% phoneme unit error rate on Asante Twi and 12.4% on Fante")
    ap.add_argument("--rows-per-file", type=int, default=40_000)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.shards).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no result shards in {args.shards}")

    KEEP = ["id", "audio_path", "shard", "row", "duration", "text", "ipa", "ipa_units",
            "n_units"]
    by_split: dict[str, list[pa.Table]] = {}
    for f in files:
        split = "test" if f.name.startswith("test") else "train"
        t = pq.read_table(f)
        by_split.setdefault(split, []).append(t.select([c for c in KEEP if c in t.column_names]))

    outdir = Path(args.out)
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    stats, empties, dup_note = {}, {}, []
    all_ids: Counter = Counter()
    for split, tables in sorted(by_split.items()):
        t = pa.concat_tables(tables).combine_chunks()
        t = t.sort_by("id")
        n = t.num_rows
        all_ids.update(t.column("id").to_pylist())

        secs = pc.sum(t.column("duration")).as_py() or 0.0
        n_units = pc.sum(t.column("n_units")).as_py() or 0
        empty = pc.sum(pc.equal(t.column("n_units"), 0)).as_py() or 0
        empties[split] = empty
        stats[split] = {
            "clips": n,
            "hours": round(secs / 3600, 2),
            "phoneme_units": int(n_units),
            "empty_transcriptions": int(empty),
            "mean_units_per_clip": round(n_units / max(n, 1), 1),
        }

        for i in range(0, n, args.rows_per_file):
            part = i // args.rows_per_file
            total = (n + args.rows_per_file - 1) // args.rows_per_file
            pq.write_table(
                t.slice(i, args.rows_per_file),
                outdir / "data" / f"{split}-{part:05d}-of-{total:05d}.parquet",
                compression="zstd",
            )
        print(f"{split}: {n} clips, {secs/3600:.2f} h, {n_units} units, {empty} empty")

    dups = [i for i, c in all_ids.items() if c > 1]
    if dups:
        dup_note.append(f"{len(dups)} duplicate ids, e.g. {dups[:5]}")
        print(f"WARNING: {len(dups)} duplicate ids — the join key is not unique")

    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))

    rows = ["| split | clips | hours | phoneme units | mean units/clip |",
            "|---|---|---|---|---|"]
    for s, v in stats.items():
        rows.append(f"| `{s}` | {v['clips']:,} | {v['hours']:,} | "
                    f"{v['phoneme_units']:,} | {v['mean_units_per_clip']} |")
    if any(empties.values()):
        rows.append("")
        rows.append("Clips the model returned nothing for: "
                    + ", ".join(f"{s} {n}" for s, n in empties.items() if n))

    repo = args.repo or "<this repo>"
    (outdir / "README.md").write_text(CARD.format(
        src=args.src, src_url=f"https://huggingface.co/datasets/{args.src}",
        model_url=f"https://huggingface.co/{args.model}", repo=repo,
        stats_table="\n".join(rows), code_url=args.code_url, uer_note=args.uer_note,
    ))
    print(f"\nwrote {outdir}")

    if args.push:
        if not args.repo:
            raise SystemExit("--push needs --repo")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(outdir), repo_id=args.repo, repo_type="dataset",
                          commit_message="IPA phoneme sidecar for new-twi-tts-aligned")
        print(f"pushed https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
