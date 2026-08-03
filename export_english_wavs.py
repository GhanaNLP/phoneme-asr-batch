"""Export the selected English clips into the shared TTS wav directory and manifest.

The Twi side carried its phonemes, speaker and QC in one parquet. The English side has them
spread across three side tables (phonemes from phonemise.py, speakers from speaker_labels.py,
selection from english_select.py), so this joins them by id, writes only the selected clips, and
appends to the existing manifest with a `language` column.

English is 16 kHz where Twi is 24 kHz, and both are resampled to a common 22.05 kHz. Upsampling
invents no high frequencies: the English voices stay band-limited to 8 kHz. That is not fixable
here and it is not fatal either — bandwidth tracks speaker identity, and the models condition on
a speaker embedding, so an English voice will simply sound like an 8 kHz speaker rather than
smearing the Twi voices. It does mean no English voice can be made full-band.
"""
from __future__ import annotations

import argparse
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

FIELDS = ["id", "speaker", "split", "duration", "text", "ipa", "n_units", "language"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/eng/data/data")
    ap.add_argument("--phonemes", default="out/eng_shards")
    ap.add_argument("--selection", default="out/eng_selection.parquet")
    ap.add_argument("--out", default="/mnt/volume_d2wey28/projects/tts-twi/data22k")
    ap.add_argument("--audio-col", default="bytes")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--language", default="eng")
    ap.add_argument("--twi-language", default="twi")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    sel = pq.read_table(args.selection,
                        columns=["id", "selected", "speaker", "duration", "text", "ipa",
                                 "n_units"]).to_pydict()
    keep = {sel["id"][i] for i in range(len(sel["id"])) if sel["selected"][i]}
    meta = {sel["id"][i]: {"speaker": sel["speaker"][i], "text": sel["text"][i],
                           "ipa": sel["ipa"][i], "n_units": sel["n_units"][i]}
            for i in range(len(sel["id"])) if sel["selected"][i]}
    print(f"{len(keep)} selected English clips")

    out = Path(args.out)
    wavdir = out / "wav"
    wavdir.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=args.threads)

    # Deterministic val split by id hash, so it is stable across reruns.
    def is_val(cid: str) -> bool:
        return (hash(cid) % 10_000) / 10_000.0 < args.val_frac

    rows: list[dict] = []
    for f in sorted(Path(args.data).glob("*.parquet")):
        t = pq.read_table(f, columns=[args.audio_col]).to_pydict()[args.audio_col]
        stem = f.stem
        idx = [i for i in range(len(t)) if f"{stem}_{i:06d}" in keep]
        if not idx:
            continue

        def one(i):
            cid = f"{stem}_{i:06d}"
            raw = t[i]["bytes"] if isinstance(t[i], dict) else t[i]
            w, in_sr = sf.read(io.BytesIO(raw), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if in_sr != args.sr:
                w = torchaudio.functional.resample(torch.from_numpy(w), in_sr,
                                                   args.sr).numpy()
            peak = float(np.abs(w).max()) if w.size else 0.0
            if peak > 1.0:
                w = w / peak
            sf.write(wavdir / f"{cid}.wav", w, args.sr, subtype="PCM_16")
            m = meta[cid]
            return {"id": cid, "speaker": m["speaker"],
                    "split": "test" if is_val(cid) else "train",
                    "duration": round(len(w) / args.sr, 3), "text": m["text"],
                    "ipa": m["ipa"], "n_units": m["n_units"], "language": args.language}

        rows.extend(pool.map(one, idx))
        print(f"  {f.name}: +{len(idx)} (total {len(rows)})", flush=True)

    # Rewrite the manifest: Twi rows gain a language column, English rows are appended.
    mpath = out / "manifest.tsv"
    old = []
    if mpath.exists():
        with open(mpath, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                    escapechar="\\"):
                lang = r.get("language") or args.twi_language
                # Drop any rows this script added before: reruns (new thresholds, a bigger
                # target) must replace the previous selection, not stack on top of it.
                if lang == args.language:
                    continue
                r["language"] = lang
                old.append({k: r.get(k, "") for k in FIELDS})
        print(f"kept {len(old)} existing non-{args.language} manifest rows")
    combined = old + [{k: r[k] for k in FIELDS} for r in rows]

    with open(mpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(combined)

    with open(out / "manifest_val.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows([r for r in combined if r["split"] == "test"])

    by_lang: dict[str, list[float]] = {}
    for r in combined:
        by_lang.setdefault(r["language"], []).append(float(r["duration"]))
    print(f"\nmanifest: {len(combined)} clips")
    for lang, ds_ in sorted(by_lang.items()):
        print(f"  {lang:5} {len(ds_):7} clips, {sum(ds_)/3600:7.1f} h, "
              f"mean {sum(ds_)/len(ds_):.1f} s")
    spk = {r["speaker"] for r in combined}
    print(f"  speakers: {len(spk)}")


if __name__ == "__main__":
    main()
