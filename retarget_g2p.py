"""Re-derive TTS targets from text via G2P, so training and inference use the same function.

The first run trained on ASR-derived phonemes and synthesised from G2P-derived ones. Measured
full-unit disagreement between those two: **0.261 for Twi, 0.509 for English** — and the TTS
round-trip error tracked it almost exactly (25.6% vs 68.6%). The model tolerates ~26% front-end
noise and falls apart at ~51%. Training on the same function used at inference removes the term
entirely.

Two deliberate choices:

**English gets canonical espeak IPA, not the Ghanaian fold.** Folding θ→t and æ→a at the symbol
level was a mistake: it made `think` and `tink` identical, so the model could never distinguish
them, and it discarded the pretrained English knowledge in the LibriTTS-R checkpoint whose
embeddings are espeak IPA. Accent is acoustic, not symbolic — write the canonical phoneme and
let the model render it with the accent that is actually in the audio.

**Existing phoneme ids are preserved.** New symbols take free slots (we used 178 of 256), so
the 12-hour checkpoint stays loadable and Twi, the speaker table and the vocoder all survive.
`graft_piper_embeddings.py` then seeds each new symbol from LibriTTS-R's learned vector for that
same phoneme, so English phonetics is transferred rather than learned from scratch.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PAD, BOS, EOS = "_", "^", "$"


def tokenize(ipa: str, symbols: list[str]) -> list[str]:
    """Greedy longest-match against the symbol set.

    Needed because a few symbols are multi-character (the diphthongs aɪ aʊ ɔɪ eɪ oʊ, and Twi's
    kʰ, t͡ʃ, k͡p). Splitting by codepoint would shatter them.
    """
    order = sorted(symbols, key=len, reverse=True)
    out: list[str] = []
    i = 0
    while i < len(ipa):
        for s in order:
            if ipa.startswith(s, i):
                out.append(s)
                i += len(s)
                break
        else:
            i += 1  # unknown codepoint (whitespace, stray mark): drop it
    return out


def espeak_ipa(text: str, voice: str) -> str:
    r = subprocess.run(["espeak-ng", "-q", "--ipa", "-v", voice, "--", text],
                       capture_output=True, text=True)
    return " ".join(r.stdout.split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/volume_d2wey28/projects/tts-twi/data22k")
    ap.add_argument("--old-map", default=None,
                    help="existing phonemes.json; its ids are preserved")
    ap.add_argument("--piper-src",
                    default="/mnt/volume_d2wey28/projects/tts-twi/piper1-gpl/src")
    ap.add_argument("--espeak-voice", default="en-us")
    ap.add_argument("--dialect", default="Asante Twi")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    old = json.loads(Path(args.old_map or data / "phonemes.json").read_text())
    used = {i for v in old.values() for i in v}
    print(f"existing map: {len(old)} symbols, max id {max(used)}")

    import ast
    src = (Path(args.piper_src) / "piper" / "phoneme_ids.py").read_text()
    piper_map = None
    for n in ast.parse(src).body:
        tg = getattr(n, "targets", None) or ([n.target] if hasattr(n, "target") else [])
        if any(getattr(t, "id", None) == "DEFAULT_PHONEME_ID_MAP" for t in tg):
            piper_map = {k: list(v) for k, v in ast.literal_eval(n.value).items()}
            break
    assert piper_map, "could not read Piper's default map"

    with open(data / "manifest.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                   escapechar="\\"))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} utterances")

    # ---- phonemise ----
    from ghana_g2p import GhanaG2P
    tw = GhanaG2P(args.dialect)
    twi_syms = set(old)  # ghana-g2p units are already in the trained map

    def do_twi(r):
        try:
            return [u for u in tw.ipa(r["text"], sep=" ", punctuation=True).split(" ") if u]
        except Exception:
            return []

    def do_eng(r):
        return espeak_ipa(r["text"], args.espeak_voice)

    pool = ThreadPoolExecutor(max_workers=args.threads)
    twi_rows = [r for r in rows if r["language"] == "twi"]
    eng_rows = [r for r in rows if r["language"] == "eng"]

    print("phonemising Twi via ghana-g2p ...", flush=True)
    twi_units = list(pool.map(do_twi, twi_rows))
    print("phonemising English via espeak-ng ...", flush=True)
    eng_ipa = list(pool.map(do_eng, eng_rows))

    # ---- extend the symbol set with whatever appeared, keeping old ids fixed ----
    new_map = {k: list(v) for k, v in old.items()}
    espeak_syms = sorted({s for s in piper_map if s not in (" ",)}, key=len, reverse=True)
    eng_units = [tokenize(s, espeak_syms) for s in eng_ipa]

    fresh: list[str] = []
    for seq in eng_units + twi_units:
        for u in seq:
            if u not in new_map:
                fresh.append(u)
    fresh = sorted(set(fresh))

    nxt = 0
    for s in fresh:
        while nxt in used:
            nxt += 1
        if nxt >= 256:
            raise SystemExit(f"out of id space; {len(fresh)} new symbols needed")
        new_map[s] = [nxt]
        used.add(nxt)
    grafted = [s for s in fresh if s in piper_map]
    print(f"added {len(fresh)} symbols: {fresh}")
    print(f"  {len(grafted)} exist in Piper's map and can be grafted from LibriTTS-R")
    print(f"  max id now {max(used)}")

    (data / "phonemes_g2p.json").write_text(json.dumps(new_map, ensure_ascii=False,
                                                       indent=1))
    graft = {s: {"new_id": new_map[s][0], "piper_id": piper_map[s][0]} for s in grafted}
    (data / "graft.json").write_text(json.dumps(graft, ensure_ascii=False, indent=1))

    # ---- write metadata ----
    def encode(units: list[str], lang: str) -> str | None:
        if not units or any(u not in new_map for u in units):
            return None
        ids = list(new_map[BOS]) + list(new_map[PAD])
        sym = f"«{lang}»"
        if sym in new_map:
            ids += list(new_map[sym]) + list(new_map[PAD])
        for u in units:
            ids += list(new_map[u]) + list(new_map[PAD])
        ids += list(new_map[EOS])
        return " ".join(map(str, ids))

    import re
    out: dict[str, list[str]] = {"train": [], "val": []}
    dropped = 0
    for rs, us in ((twi_rows, twi_units), (eng_rows, eng_units)):
        for r, u in zip(rs, us):
            enc = encode(u, r["language"])
            if enc is None:
                dropped += 1
                continue
            text = re.sub(r'[|"\r\n\t]+', " ", r["text"] or "").strip()
            line = f"{r['id']}.wav|{r['speaker']}|{text}|{enc}"
            out["val" if r["split"] == "test" else "train"].append(line)

    for name, key in (("train", "train"), ("val", "val")):
        p = data / f"metadata_{name}_g2p.csv"
        p.write_text("\n".join(out[key]) + "\n", encoding="utf-8")
        print(f"metadata_{name}_g2p.csv: {len(out[key])} rows")
    if dropped:
        print(f"dropped {dropped} rows with unmappable phonemes")

    # A quick look at what changed, so the shift is visible rather than assumed.
    for lbl, rs, us in (("twi", twi_rows, twi_units), ("eng", eng_rows, eng_units)):
        if rs:
            print(f"\n{lbl} example:\n  text {rs[0]['text'][:80]}")
            print(f"  old  {' '.join((rs[0]['ipa'] or '').split(' ')[:20])}")
            print(f"  new  {' '.join(us[0][:20])}")


if __name__ == "__main__":
    main()
