"""Merge per-shard phoneme results into a publishable dataset and optionally push it.

Two shapes, chosen with --with-audio:

  sidecar (default)  no audio, one row per clip keyed by the source clip filename. A few MB.
                     Users load the source audio dataset and this one and join on `id`.
  full               the source audio carried alongside the phonemes, so the result is
                     self-contained and trains a TTS model without a join step.

In full mode the audio bytes are copied straight across from the source parquet — never
decoded and re-encoded. Re-encoding would be slow, lossy for a lossy source, and would make
the published waveforms differ from the ones the phonemes were actually derived from.

Rows are matched on (shard, row), which `phonemise.py` records for exactly this purpose, and
the source `audio.path` is asserted against the recorded one so a mismatched or reordered
source shard fails loudly instead of silently pairing the wrong clip with the wrong phonemes.

Splits come from the source shard filenames, so the train/test boundary matches the source
dataset by construction rather than by assumption.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HEADER = """---
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
"""

CARD = """
# new-twi-tts-aligned + IPA phonemes

[`ghanaopendata/new-twi-tts-aligned`]({src_url}) with a machine-generated **IPA phoneme
transcription for every clip**, produced with
[`ghananlpcommunity/ghana-speech-phoneme-asr`]({model_url}).

{intro}

## Contents

{stats_table}

## Columns

{columns}

## Loading it

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train")
row = ds[0]
row["audio"]["array"]      # 24 kHz mono waveform
row["text"]                # orthographic Twi
row["ipa"]                 # "n a n s o pʰ e tʰ o ɾ o ..."
row["ipa"].split(" ")      # phoneme units — see the warning below
```

## Read `ipa`, not the characters

Many units are multi-character — `kʰ`, `t͡ʃ`, `k͡p`, `hʷ`, `iː`. The inventory is 172 units,
not 172 characters. **Split on spaces:**

```python
units = row["ipa"].split(" ")     # correct
units = list(row["ipa"])          # wrong — tears k͡p into three characters
```

## How it was made

Greedy CTC decoding of the fairseq2 checkpoint in bf16 on a single H200, in length-sorted
batches — 172 hours in about 14 minutes (~780x realtime). Code, including the validation
harness that checks the audio front-end against the reference decoder:
{code_url}

{audio_note}

## Accuracy, and what to expect

These are **model predictions, not verified ground truth.** On its own held-out dev set the
model scores {uer_note}. This is a strong starting point for TTS phoneme targets or
pronunciation analysis, not a gold lexicon.

Three properties worth knowing before training on it:

- **Punctuation is guessed.** The model emits punctuation marks, but they have no acoustic
  realisation, so its punctuation error rate is high (~30%) even where the phonemes are
  good. Filter against a known punctuation set if you do not want it.
- **The IPA follows the speech, not the spelling.** Where a speaker elides or reduces, the
  phonemes reflect what was said and will not match a rule-based grapheme-to-phoneme
  rendering of `text`. That is the reason to use an acoustic model — and the reason `ipa`
  and `text` will legitimately disagree.
- **Batching adds ~0.2% unit error** versus decoding each clip alone, because wav2vec2's
  convolutional front-end is not padding-masked. Far below the model's own error rate.

{empty_note}

## License

`cc-by-nc-4.0`, inherited from the source audio dataset.
"""

SIDECAR_COLS = """| column | type | meaning |
|---|---|---|
| `id` | string | clip id, e.g. `segment_042811` — the join key |
| `audio_path` | string | source `audio.path`, e.g. `segment_042811.wav` |
| `shard` | string | source parquet shard the clip came from |
| `row` | int32 | row index within that shard |
| `duration` | float32 | clip length in seconds |
| `text` | string | source orthographic Twi text, unchanged |
| `ipa` | string | phonemes, **space-separated** |
| `ipa_units` | list[string] | the same phonemes as a list |
| `n_units` | int32 | number of phoneme units |"""

FULL_COLS = """| column | type | meaning |
|---|---|---|
| `id` | string | clip id, e.g. `segment_042811` |
| `audio` | Audio | 24 kHz mono, copied bit-for-bit from the source dataset |
| `text` | string | source orthographic Twi text, unchanged |
| `ipa` | string | phonemes, **space-separated** |
| `ipa_units` | list[string] | the same phonemes as a list |
| `duration` | float32 | clip length in seconds |
| `n_units` | int32 | number of phoneme units |"""


def hf_features(with_audio: bool) -> dict:
    """The `huggingface` schema metadata that makes `load_dataset` type these columns."""
    f: dict = {"id": {"dtype": "string", "_type": "Value"}}
    if with_audio:
        f["audio"] = {"sampling_rate": 24000, "mono": True, "decode": True, "_type": "Audio"}
    f.update({
        "text": {"dtype": "string", "_type": "Value"},
        "ipa": {"dtype": "string", "_type": "Value"},
        "ipa_units": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "duration": {"dtype": "float32", "_type": "Value"},
        "n_units": {"dtype": "int32", "_type": "Value"},
    })
    return {"info": {"features": f}}


def build_full(res: pa.Table, src_shard: Path, audio_col: str) -> pa.Table:
    """Attach phonemes to the source shard's audio, matching on recorded row order."""
    src = pq.read_table(src_shard, columns=[audio_col])
    if src.num_rows != res.num_rows:
        raise SystemExit(
            f"{src_shard.name}: source has {src.num_rows} rows, results have {res.num_rows}")

    res = res.sort_by("row")  # results were written in source order; make that explicit
    if res.column("row").to_pylist() != list(range(res.num_rows)):
        raise SystemExit(f"{src_shard.name}: result rows are not a full 0..n-1 range")

    # The audio must line up with the phonemes. Check, do not hope.
    want = res.column("audio_path").to_pylist()
    got = [a["path"] for a in src.column(audio_col).to_pylist()]
    if want != got:
        bad = next(i for i, (a, b) in enumerate(zip(want, got)) if a != b)
        raise SystemExit(f"{src_shard.name}: audio/phoneme mismatch at row {bad}: "
                         f"results say {want[bad]}, source says {got[bad]}")

    t = pa.table({
        "id": res.column("id"),
        "audio": src.column(audio_col),
        "text": res.column("text"),
        "ipa": res.column("ipa"),
        "ipa_units": res.column("ipa_units"),
        "duration": res.column("duration"),
        "n_units": res.column("n_units"),
    })
    return t.replace_schema_metadata({"huggingface": json.dumps(hf_features(True))})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="out/shards", help="phonemise.py results")
    ap.add_argument("--out", default="out/dataset")
    ap.add_argument("--with-audio", action="store_true",
                    help="carry the source audio into the output (self-contained, large)")
    ap.add_argument("--source-shards", default="data/raw/data",
                    help="the source parquet shards, required with --with-audio")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--src", default="ghanaopendata/new-twi-tts-aligned")
    ap.add_argument("--model", default="ghananlpcommunity/ghana-speech-phoneme-asr")
    ap.add_argument("--repo", default=None, help="HF dataset repo id to push to")
    ap.add_argument("--code-url", default="https://github.com/GhanaNLP/phoneme-asr-batch")
    ap.add_argument("--uer-note",
                    default="17.1% phoneme unit error rate on Asante Twi and 12.4% on Fante")
    ap.add_argument("--rows-per-file", type=int, default=40_000,
                    help="sidecar mode only; full mode mirrors the source shards")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--clear-remote", action="store_true",
                    help="delete data/ in the target repo first (use when changing shape)")
    args = ap.parse_args()

    files = sorted(Path(args.shards).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no result shards in {args.shards}")

    outdir = Path(args.out)
    (outdir / "data").mkdir(parents=True, exist_ok=True)

    SIDECAR_KEEP = ["id", "audio_path", "shard", "row", "duration", "text", "ipa",
                    "ipa_units", "n_units"]
    stats: dict = {}
    empties: dict = {}
    all_ids: Counter = Counter()
    acc: dict[str, list[pa.Table]] = {}

    def tally(split: str, t: pa.Table) -> None:
        s = stats.setdefault(split, {"clips": 0, "seconds": 0.0, "phoneme_units": 0})
        s["clips"] += t.num_rows
        s["seconds"] += pc.sum(t.column("duration")).as_py() or 0.0
        s["phoneme_units"] += int(pc.sum(t.column("n_units")).as_py() or 0)
        empties[split] = empties.get(split, 0) + int(
            pc.sum(pc.equal(t.column("n_units"), 0)).as_py() or 0)
        all_ids.update(t.column("id").to_pylist())

    for f in files:
        split = "test" if f.name.startswith("test") else "train"
        res = pq.read_table(f)

        if args.with_audio:
            # Mirror the source sharding: the source shards are already ~500 MB, which is
            # the size the Hub wants, and it keeps peak memory to one shard.
            src_shard = Path(args.source_shards) / f.name
            if not src_shard.exists():
                raise SystemExit(f"missing source shard {src_shard}")
            t = build_full(res, src_shard, args.audio_col)
            pq.write_table(t, outdir / "data" / f.name, compression="zstd")
            tally(split, t)
            print(f"{f.name:34} {t.num_rows:6} clips  "
                  f"{(outdir / 'data' / f.name).stat().st_size/1e6:7.0f} MB", flush=True)
        else:
            acc.setdefault(split, []).append(
                res.select([c for c in SIDECAR_KEEP if c in res.column_names]))

    if not args.with_audio:
        for split, tables in sorted(acc.items()):
            t = pa.concat_tables(tables).combine_chunks().sort_by("id")
            tally(split, t)
            n = t.num_rows
            total = (n + args.rows_per_file - 1) // args.rows_per_file
            meta = {"huggingface": json.dumps(hf_features(False))}
            for i in range(0, n, args.rows_per_file):
                pq.write_table(
                    t.slice(i, args.rows_per_file).replace_schema_metadata(meta),
                    outdir / "data" / f"{split}-{i//args.rows_per_file:05d}-of-{total:05d}.parquet",
                    compression="zstd")

    for s, v in stats.items():
        v["hours"] = round(v["seconds"] / 3600, 2)
        v["mean_units_per_clip"] = round(v["phoneme_units"] / max(v["clips"], 1), 1)
        v["empty_transcriptions"] = empties.get(s, 0)
        del v["seconds"]
        print(f"{s}: {v['clips']} clips, {v['hours']} h, {v['phoneme_units']} units, "
              f"{v['empty_transcriptions']} empty")

    dups = [i for i, c in all_ids.items() if c > 1]
    if dups:
        print(f"WARNING: {len(dups)} duplicate ids, e.g. {dups[:5]}")

    (outdir / "stats.json").write_text(json.dumps(stats, indent=2))

    rows = ["| split | clips | hours | phoneme units | mean units/clip |",
            "|---|---|---|---|---|"]
    for s, v in sorted(stats.items()):
        rows.append(f"| `{s}` | {v['clips']:,} | {v['hours']:,} | "
                    f"{v['phoneme_units']:,} | {v['mean_units_per_clip']} |")

    n_empty = sum(empties.values())
    repo = args.repo or "<this repo>"
    (outdir / "README.md").write_text(HEADER + CARD.format(
        src=args.src, src_url=f"https://huggingface.co/datasets/{args.src}",
        model_url=f"https://huggingface.co/{args.model}", repo=repo,
        stats_table="\n".join(rows), code_url=args.code_url, uer_note=args.uer_note,
        columns=FULL_COLS if args.with_audio else SIDECAR_COLS,
        intro=("Audio included — this is self-contained, no join with the source dataset "
               "needed." if args.with_audio else
               "This is a **sidecar dataset: it contains no audio.** It is keyed by the "
               "source clip filename so you can attach phonemes without a second copy of "
               "the waveforms."),
        audio_note=("The audio is copied bit-for-bit from the source dataset — never decoded "
                    "and re-encoded — so these are exactly the waveforms the phonemes were "
                    "derived from." if args.with_audio else ""),
        empty_note=(f"The model returned no phonemes for {n_empty} clips "
                    f"(`n_units == 0`); filter them out if that matters."
                    if n_empty else ""),
    ))
    print(f"\nwrote {outdir}")

    if args.push:
        if not args.repo:
            raise SystemExit("--push needs --repo")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", exist_ok=True)
        if args.clear_remote:
            # Shape changes rename the shards; without this the old files stay and the
            # data_files globs would match both sets.
            api.delete_folder("data", repo_id=args.repo, repo_type="dataset",
                              commit_message="drop previous data files")
            print("cleared remote data/")
        if args.with_audio:
            api.upload_large_folder(folder_path=str(outdir), repo_id=args.repo,
                                    repo_type="dataset")
        else:
            api.upload_folder(folder_path=str(outdir), repo_id=args.repo,
                              repo_type="dataset",
                              commit_message="IPA phonemes for new-twi-tts-aligned")
        print(f"pushed https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
