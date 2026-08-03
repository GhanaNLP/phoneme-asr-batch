"""Attach pseudo-speaker labels and QC flags to the phonemised dataset.

Nothing is deleted. Bad rows are marked, not dropped, because "bad" here is a judgement call
that depends on the architecture — a 25-second clip is unusable for Piper and fine for F5-TTS.
Training scripts filter on `qc_pass`; anyone who disagrees with a rule can ignore it and use
`qc_reason` to make their own cut.

The rules, and why each one exists:

  no_phonemes    the ASR returned nothing. Untrainable.
  too_short      under 0.4 s. Mostly clipped fragments; TTS aligners choke on them.
  too_long       over 20 s. The long tail, and above most of these frameworks' memory budget.
  bad_rate       phoneme rate outside 4-25 units/s. The corpus median is 11.4 and p99 is 18.5,
                 so this is a wide band — it is catching ASR failure (silence decoded as a
                 handful of units, or a hallucinated run), not natural speed variation.
  short_vs_text  fewer than 0.35 phonemes per orthographic character, where the median is
                 0.74. Means the ASR dropped most of the utterance.
  rare_speaker   fewer than 20 clips for its pseudo-speaker. A speaker embedding cannot be
                 learned from a handful of clips, and these are also where the clustering is
                 least trustworthy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

MIN_DUR, MAX_DUR = 0.4, 20.0
MIN_RATE, MAX_RATE = 4.0, 25.0
MIN_UNITS_PER_CHAR = 0.35
MIN_SPEAKER_CLIPS = 20


def hf_features() -> dict:
    return {"info": {"features": {
        "id": {"dtype": "string", "_type": "Value"},
        "audio": {"sampling_rate": 24000, "mono": True, "decode": True, "_type": "Audio"},
        "text": {"dtype": "string", "_type": "Value"},
        "ipa": {"dtype": "string", "_type": "Value"},
        "ipa_units": {"feature": {"dtype": "string", "_type": "Value"}, "_type": "Sequence"},
        "duration": {"dtype": "float32", "_type": "Value"},
        "n_units": {"dtype": "int32", "_type": "Value"},
        "speaker": {"dtype": "string", "_type": "Value"},
        "speaker_idx": {"dtype": "int32", "_type": "Value"},
        "qc_pass": {"dtype": "bool", "_type": "Value"},
        "qc_reason": {"dtype": "string", "_type": "Value"},
    }}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/dataset_full/data")
    ap.add_argument("--speakers", default="out/speakers_0.70.parquet")
    ap.add_argument("--out", default="out/dataset_ready")
    args = ap.parse_args()

    spk = pq.read_table(args.speakers).to_pydict()
    spk_of = dict(zip(spk["id"], spk["speaker"]))
    idx_of = dict(zip(spk["id"], spk["speaker_idx"]))

    counts: dict[str, int] = {}
    for s in spk["speaker"]:
        counts[s] = counts.get(s, 0) + 1
    small = {s for s, c in counts.items() if c < MIN_SPEAKER_CLIPS}
    print(f"{len(counts)} pseudo-speakers, {len(small)} under {MIN_SPEAKER_CLIPS} clips")

    outdir = Path(args.out) / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"huggingface": json.dumps(hf_features())}

    tally: dict[str, int] = {}
    kept = total = 0
    kept_secs = 0.0

    for f in sorted(Path(args.data).glob("*.parquet")):
        t = pq.read_table(f)
        ids = t.column("id").to_pylist()
        dur = t.column("duration").to_pylist()
        nu = t.column("n_units").to_pylist()
        txt = t.column("text").to_pylist()

        speakers, idxs, passes, reasons = [], [], [], []
        for i, cid in enumerate(ids):
            s = spk_of.get(cid)
            speakers.append(s or "spk_unknown")
            idxs.append(idx_of.get(cid, -1))

            why = []
            if nu[i] == 0:
                why.append("no_phonemes")
            if dur[i] < MIN_DUR:
                why.append("too_short")
            if dur[i] > MAX_DUR:
                why.append("too_long")
            rate = nu[i] / max(dur[i], 1e-3)
            if not (MIN_RATE <= rate <= MAX_RATE):
                why.append("bad_rate")
            if len(txt[i] or "") and nu[i] / max(len(txt[i]), 1) < MIN_UNITS_PER_CHAR:
                why.append("short_vs_text")
            if s is None or s in small:
                why.append("rare_speaker")

            ok = not why
            passes.append(ok)
            reasons.append("" if ok else ",".join(why))
            for w in why:
                tally[w] = tally.get(w, 0) + 1
            total += 1
            if ok:
                kept += 1
                kept_secs += dur[i]

        out = pa.table({
            "id": t.column("id"), "audio": t.column("audio"), "text": t.column("text"),
            "ipa": t.column("ipa"), "ipa_units": t.column("ipa_units"),
            "duration": t.column("duration"), "n_units": t.column("n_units"),
            "speaker": pa.array(speakers, pa.string()),
            "speaker_idx": pa.array(idxs, pa.int32()),
            "qc_pass": pa.array(passes, pa.bool_()),
            "qc_reason": pa.array(reasons, pa.string()),
        }).replace_schema_metadata(meta)
        pq.write_table(out, outdir / f.name, compression="zstd")

    print(f"\nkept {kept}/{total} ({kept/total:.1%}), {kept_secs/3600:.1f} h trainable")
    print("flagged (rows can carry several reasons):")
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"  {k:16} {v:7} ({v/total:5.2%})")

    Path(args.out, "qc_stats.json").write_text(json.dumps(
        {"kept": kept, "total": total, "kept_hours": round(kept_secs / 3600, 2),
         "flags": tally, "speakers": len(counts),
         "speakers_used": len(counts) - len(small)}, indent=2))


if __name__ == "__main__":
    main()
