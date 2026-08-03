"""Turn the phonemised parquet dataset into the on-disk layouts TTS trainers expect.

Almost every light TTS trainer wants the same two things: a directory of wav files and a
manifest. Decoding 151k clips is the slow part, so `wavs` does it once at a chosen sample rate
and the per-framework subcommands only write text files against that shared directory.

    tts_data.py wavs   --sr 22050          # once; ~150k wavs + manifest.tsv
    tts_data.py piper                      # metadata.csv + phonemes.json

22.05 kHz is the default because the VITS family (Piper, Coqui VITS, MeloTTS, VITS2) and the
acoustic models (Matcha, Glow-TTS, FastSpeech2) all ship 22.05 kHz pretrained checkpoints, and
finetuning from one beats training from scratch by a wide margin. F5-TTS and StyleTTS2 want
24 kHz — run `wavs --sr 24000 --out ...` again for those.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio

# Piper's own specials, reused so ids stay compatible with its pretrained checkpoints.
PAD, BOS, EOS = "_", "^", "$"


def export_wavs(data: Path, out: Path, sr: int, threads: int, qc_only: bool) -> None:
    wavdir = out / "wav"
    wavdir.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=threads)
    rows: list[dict] = []
    written = skipped = 0

    for f in sorted(data.glob("*.parquet")):
        t = pq.read_table(f).to_pydict()
        split = "test" if f.name.startswith("test") else "train"

        def one(i):
            if qc_only and not t["qc_pass"][i]:
                return None
            w, in_sr = sf.read(io.BytesIO(t["audio"][i]["bytes"]), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            if in_sr != sr:
                w = torchaudio.functional.resample(torch.from_numpy(w), in_sr, sr).numpy()
            # Guard against resampler overshoot before quantising to 16-bit.
            peak = float(np.abs(w).max()) if w.size else 0.0
            if peak > 1.0:
                w = w / peak
            sf.write(wavdir / f"{t['id'][i]}.wav", w, sr, subtype="PCM_16")
            return {
                "id": t["id"][i], "speaker": t["speaker"][i], "split": split,
                "duration": round(len(w) / sr, 3), "text": t["text"][i],
                "ipa": t["ipa"][i], "n_units": t["n_units"][i],
            }

        for r in pool.map(one, range(len(t["id"]))):
            if r is None:
                skipped += 1
            else:
                rows.append(r)
                written += 1
        print(f"  {f.name}: {written} written, {skipped} skipped", flush=True)

    with open(out / "manifest.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(rows)

    spk = sorted({r["speaker"] for r in rows})
    secs = sum(r["duration"] for r in rows)
    print(f"\n{written} wavs at {sr} Hz, {secs/3600:.1f} h, {len(spk)} speakers")
    print(f"manifest: {out / 'manifest.tsv'}")


def read_manifest(out: Path) -> list[dict]:
    with open(out / "manifest.tsv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                   escapechar="\\"))


def build_piper_map(units: list[str], piper_src: Path | None) -> dict[str, list[int]]:
    """Map our IPA units onto Piper's ids, reusing its id for every symbol it already knows.

    This is the whole trick for a good warm start. Piper's default map is espeak IPA, so most
    of our inventory (a, b, e, k, m, n, o, s, t, u, ŋ, ç, ɔ, ɛ ...) already has an id whose
    pretrained embedding row means the same sound. Matching those ids means finetuning starts
    from real phonetic knowledge instead of random vectors, and only the genuinely new units
    (k͡p, t͡ʃ, kʰ, hʷ ...) have to be learned from scratch.
    """
    default: dict[str, list[int]] = {}
    if piper_src is not None:
        # Read the literal out of the source rather than importing it: piper/__init__.py
        # pulls in onnxruntime, which this script has no other use for.
        import ast
        src = (piper_src / "piper" / "phoneme_ids.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in tree.body:
            targets = getattr(node, "targets", []) or ([node.target] if
                                                       hasattr(node, "target") else [])
            if any(getattr(t, "id", None) == "DEFAULT_PHONEME_ID_MAP" for t in targets):
                default = {k: list(v) for k, v in ast.literal_eval(node.value).items()}
                break
        if not default:
            raise SystemExit("could not find DEFAULT_PHONEME_ID_MAP in piper source")
        print(f"loaded Piper default map: {len(default)} symbols")

    id_map: dict[str, list[int]] = {
        PAD: default.get(PAD, [0]), BOS: default.get(BOS, [1]),
        EOS: default.get(EOS, [2]), " ": default.get(" ", [3]),
    }
    used = {i for v in id_map.values() for i in v}
    reused = []
    for u in units:
        if u in id_map:
            continue
        if u in default and not (set(default[u]) & used):
            id_map[u] = list(default[u])
            used.update(id_map[u])
            reused.append(u)

    nxt = 0
    for u in units:
        if u in id_map:
            continue
        while nxt in used:
            nxt += 1
        if nxt >= 256:
            raise SystemExit("ran out of ids under Piper's 256-symbol default")
        id_map[u] = [nxt]
        used.add(nxt)

    print(f"phoneme map: {len(id_map)} symbols, {len(reused)} reused Piper ids "
          f"(warm start), {len(units) - len(reused)} new, max id {max(used)}")
    return id_map


def export_piper(out: Path, tokens: Path, piper_src: Path | None,
                 lang_tokens: bool = True) -> None:
    vocab = []
    for line in tokens.read_text(encoding="utf-8").splitlines():
        if line:
            sym, _ = line.rsplit(" ", 1)
            vocab.append(sym)
    units = [u for u in vocab if not (u.startswith("<") and u.endswith(">"))]

    id_map = build_piper_map(units, piper_src)
    rows = read_manifest(out)

    # A language token prefixed to each sequence. The speaker embedding already correlates
    # with language here (the two speaker sets are disjoint), but the token makes language
    # an explicit control at inference, so a voice can be asked to read the other language.
    langs = sorted({r.get("language", "") for r in rows if r.get("language")})
    lang_sym: dict[str, str] = {}
    if lang_tokens and len(langs) > 1:
        nxt = max(i for v in id_map.values() for i in v) + 1
        for lang in langs:
            sym = f"«{lang}»"
            if nxt >= 256:
                raise SystemExit("no id space left for language tokens")
            id_map[sym] = [nxt]
            lang_sym[lang] = sym
            nxt += 1
        print(f"language tokens: " +
              ", ".join(f"{k}={id_map[v][0]}" for k, v in lang_sym.items()))

    (out / "phonemes.json").write_text(json.dumps(id_map, ensure_ascii=False, indent=1))
    dropped = 0
    for name, want in (("train", "train"), ("val", "test")):
        with open(out / f"metadata_{name}.csv", "w", encoding="utf-8") as fh:
            n = 0
            for r in rows:
                if r["split"] != want:
                    continue
                us = r["ipa"].split(" ") if r["ipa"] else []
                if any(u not in id_map for u in us) or not us:
                    dropped += 1
                    continue
                ids = list(id_map[BOS]) + list(id_map[PAD])
                sym = lang_sym.get(r.get("language", ""))
                if sym:
                    ids += list(id_map[sym]) + list(id_map[PAD])
                for u in us:
                    ids += list(id_map[u]) + list(id_map[PAD])
                ids += list(id_map[EOS])
                # utt_id|speaker|text|phoneme_ids   (text is kept for logging only)
                # Strip the delimiter, quote characters and newlines: the reader is a plain
                # csv.reader, so a bare double quote makes it treat the rest as a quoted
                # field and swallow subsequent lines until the next quote — which surfaces
                # much later as "field larger than field limit".
                text = re.sub(r'[|"\r\n\t]+', " ", r["text"] or "").strip()
                fh.write(f"{r['id']}.wav|{r['speaker']}|{text}|"
                         f"{' '.join(map(str, ids))}\n")
                n += 1
        print(f"metadata_{name}.csv: {n} rows")
    if dropped:
        print(f"dropped {dropped} rows with unmappable or empty phonemes")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("wavs")
    w.add_argument("--data", default="out/dataset_ready/data")
    w.add_argument("--out", default="out/tts22k")
    w.add_argument("--sr", type=int, default=22050)
    w.add_argument("--threads", type=int, default=16)
    w.add_argument("--all", action="store_true", help="include qc_pass == false rows")

    p = sub.add_parser("piper")
    p.add_argument("--out", default="out/tts22k")
    p.add_argument("--tokens",
                   default="/mnt/volume_d2wey28/projects/ghana-phoneme-asr/release/onnx/tokens.txt")
    p.add_argument("--piper-src",
                   default="/mnt/volume_d2wey28/projects/tts-twi/piper1-gpl/src")
    p.add_argument("--no-lang-tokens", action="store_true",
                   help="omit the per-language token prefix")

    a = ap.parse_args()
    if a.cmd == "wavs":
        export_wavs(Path(a.data), Path(a.out), a.sr, a.threads, not a.all)
    else:
        export_piper(Path(a.out), Path(a.tokens),
                     Path(a.piper_src) if a.piper_src else None,
                     lang_tokens=not a.no_lang_tokens)


if __name__ == "__main__":
    main()
