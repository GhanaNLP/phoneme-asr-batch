"""Score synthesised English with an English phoneme recogniser.

The Ghana phoneme ASR is the wrong instrument for English: English is not among its 42 training
languages, and real human recordings score ~70% unit error against a canonical reference. A TTS
number measured on top of a 70% floor says almost nothing.

KoelLabs/xlsr-english-01 is a wav2vec2 CTC phoneme recogniser trained on *accented* English
(L2-ARCTIC, EpaDB, Speech Ocean) at ~19% PER, and it emits IPA — so it compares like-for-like
against IPA targets instead of proxying through words. Run `convert_koel_onnx.py` first; this
loads the ONNX through sherpa-onnx, so the same artifact works on any platform.

The two inventories are both IPA but not identical, so a comparison has to normalise first or it
measures notation rather than pronunciation:

  espeak marks stress (ˈ ˌ) and length (ː); KoelLabs has no such symbols     -> drop
  KoelLabs marks nasalisation and syllabicity (ə̃, n̩, ŋ̍); espeak does not    -> strip to base
  KoelLabs marks aspiration (kʰ pʰ θʰ sʰ); espeak does not for English       -> strip
  the same vowel has different letters (ɒ/ɑ, əʊ/oʊ, ɝ/ɚ)                     -> unify
  t-flapping is allophonic and both spell it either way (ɾ/t)                -> unify

Everything left is a real pronunciation difference. As always, read the *gap* against the floor
measured on real recordings, not the absolute number: this recogniser has its own ~19% PER and
the reference text is itself imperfect.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path

# KoelLabs / espeak symbol unification. Applied to both sides, so it cannot bias one.
UNIFY = {
    "ɒ": "ɑ", "əʊ": "oʊ", "ɝ": "ɚ", "ɾ": "t", "ʔ": "", "x": "k", "β": "b",
    "ɣ": "ɡ", "ɦ": "h", "ʉ": "u", "ɨ": "ɪ", "ᵻ": "ɪ", "ɐ": "ʌ", "g": "ɡ",
    "r": "ɹ", "ɜ": "ɚ",  # espeak writes NURSE as ɜː, KoelLabs as ɝ/ɚ
}
DROP = set("ˈˌːˑ'|- ")
# Combining marks for nasalisation, syllabicity, devoicing.
COMBINING = {"̃", "̩", "̍", "̥", "̯"}


def normalize(units: list[str]) -> list[str]:
    out: list[str] = []
    for u in units:
        u = "".join(c for c in unicodedata.normalize("NFD", u)
                    if c not in COMBINING and c not in DROP)
        u = u.replace("ʰ", "").replace("͡", "")
        if not u:
            continue
        u = UNIFY.get(u, u)
        if not u:
            continue
        # Collapse runs of the same phoneme: length is not a content difference here.
        if not out or out[-1] != u:
            out.append(u)
    return out


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


def recognise(paths: list[Path], model: str, tokens: str, threads: int) -> list[list[str]]:
    import sherpa_onnx
    import soundfile as sf

    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=model, tokens=tokens, num_threads=threads)
    out = []
    for p in paths:
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(axis=1)
        s = rec.create_stream()
        s.accept_waveform(sr, w)
        rec.decode_stream(s)
        out.append(list(s.result.tokens))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="TSV with id, ipa, language")
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--real-dir", default=None, help="real recordings, to measure the floor")
    ap.add_argument("--model", default="out/koel-en/model.int8.onnx")
    ap.add_argument("--tokens", default="out/koel-en/tokens.txt")
    ap.add_argument("--language", default="eng")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    with open(args.manifest, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                          escapechar="\\")
                if r.get("language") == args.language]
    synth = Path(args.synth_dir)
    rows = [r for r in rows if (synth / f"{r['id']}.wav").exists()][: args.limit]
    if not rows:
        raise SystemExit(f"no {args.language} wavs found in {synth}")
    print(f"scoring {len(rows)} {args.language} utterances with KoelLabs xlsr-english-01")

    ref = [normalize(r["ipa"].split(" ")) for r in rows]
    hyp = [normalize(h) for h in
           recognise([synth / f"{r['id']}.wav" for r in rows], args.model, args.tokens,
                     args.threads)]

    def per(refs, hyps):
        e = sum(edit_distance(a, b) for a, b in zip(refs, hyps))
        n = sum(len(a) for a in refs)
        return e / max(n, 1)

    result = {"n": len(rows), "synth_per": round(per(ref, hyp), 4)}
    print(f"\nphoneme error on synthesised audio : {result['synth_per']:.2%}")

    if args.real_dir:
        real = Path(args.real_dir)
        idx = [i for i, r in enumerate(rows) if (real / f"{r['id']}.wav").exists()]
        if idx:
            rh = [normalize(h) for h in
                  recognise([real / f"{rows[i]['id']}.wav" for i in idx], args.model,
                            args.tokens, args.threads)]
            rr = [ref[i] for i in idx]
            result["real_per"] = round(per(rr, rh), 4)
            result["gap"] = round(result["synth_per"] - result["real_per"], 4)
            result["n_real"] = len(idx)
            print(f"phoneme error on real recordings   : {result['real_per']:.2%}"
                  f"   [n={len(idx)}]")
            print(f"gap                                : {result['gap']:+.2%}"
                  f"   <- the number to watch")

    empty = sum(1 for h in hyp if not h)
    if empty:
        print(f"\nwarning: nothing recognised in {empty} synthesised clips")

    for i in range(min(args.show, len(rows))):
        print(f"\n  {rows[i]['id']}")
        print(f"    text {rows[i].get('text','')[:90]}")
        print(f"    want {' '.join(ref[i][:34])}")
        print(f"    got  {' '.join(hyp[i][:34])}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
