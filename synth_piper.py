"""Synthesise from a Piper checkpoint using phoneme units straight from the manifest.

Piper's own `infer_torch.py` phonemises its input text, and with `phoneme_type: text` that
means one phoneme per *codepoint* — which would tear `kʰ` and `k͡p` apart. Our units are
already phonemes, so this bypasses phonemisation entirely and maps units to ids with the same
BOS/PAD/EOS convention and language token that `tts_data.py` used for training.

Output is `<id>.wav` per utterance, which is what `tts_eval.py` expects, so the round-trip
phoneme UER can be measured against the same manifest that drove the synthesis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

PAD, BOS, EOS = "_", "^", "$"


def to_ids(units: list[str], id_map: dict, lang: str | None) -> list[int] | None:
    if any(u not in id_map for u in units) or not units:
        return None
    ids = list(id_map[BOS]) + list(id_map[PAD])
    if lang:
        sym = f"«{lang}»"
        if sym in id_map:
            ids += list(id_map[sym]) + list(id_map[PAD])
    for u in units:
        ids += list(id_map[u]) + list(id_map[PAD])
    return ids + list(id_map[EOS])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="config.json written by training")
    ap.add_argument("--manifest", required=True, help="TSV with id, ipa, speaker, language")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--noise-scale", type=float, default=0.667)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--noise-w", type=float, default=0.8)
    ap.add_argument("--speaker", default=None,
                    help="force one speaker name; default uses each row's own")
    args = ap.parse_args()

    sys.path.insert(0, "/mnt/volume_d2wey28/projects/tts-twi/piper1-gpl/src")
    from piper.train.vits.lightning import VitsModel
    from piper.train.vits.utils import audio_float_to_int16

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    sr = cfg["audio"]["sample_rate"] if "audio" in cfg else cfg["sample_rate"]
    id_map = cfg["phoneme_id_map"]
    spk_map = cfg.get("speaker_id_map") or {}
    print(f"config: {sr} Hz, {len(id_map)} symbols, {len(spk_map)} speakers")

    model = VitsModel.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    with torch.no_grad():
        model.model_g.dec.remove_weight_norm()
    model = model.to(args.device)

    with open(args.manifest, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                   escapechar="\\"))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    scales = [args.noise_scale, args.length_scale, args.noise_w]

    done = skipped = 0
    for r in rows:
        if done >= args.limit:
            break
        ids = to_ids((r.get("ipa") or "").split(" "), id_map, r.get("language"))
        if ids is None:
            skipped += 1
            continue

        name = args.speaker or r.get("speaker")
        sid = spk_map.get(name)
        if sid is None and spk_map:
            skipped += 1
            continue

        t = torch.LongTensor(ids).unsqueeze(0).to(args.device)
        tl = torch.LongTensor([len(ids)]).to(args.device)
        s = torch.LongTensor([sid]).to(args.device) if sid is not None else None
        with torch.no_grad():
            audio = model(t, tl, scales, sid=s).detach().cpu().numpy()

        pcm = audio_float_to_int16(audio)
        with wave.open(str(outdir / f"{r['id']}.wav"), "wb") as w:
            w.setframerate(sr)
            w.setsampwidth(2)
            w.setnchannels(1)
            w.writeframes(pcm.tobytes())
        done += 1
        if done % 50 == 0:
            print(f"  {done} synthesised", flush=True)

    print(f"\nwrote {done} wavs to {outdir}" + (f", skipped {skipped}" if skipped else ""))


if __name__ == "__main__":
    main()
