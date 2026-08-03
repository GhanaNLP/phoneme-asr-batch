"""Score the Ghanaian English clips and select a high-quality, Twi-balanced subset.

There is ~1,068 h of English against 164 h of Twi. Training on all of it makes a mostly-English
model, so we take roughly Twi's worth. Being able to keep only ~15% is a luxury: it means the
cut can be made on quality rather than on whatever happens to be first.

The load-bearing signal is **does the audio actually say what the text says**. The phoneme ASR
gives us IPA from the audio; CMUdict gives us an expected pronunciation from the text. If they
agree, both the transcript and the recording are probably sound. If they diverge, something is
wrong — wrong text, music, crosstalk, a truncated clip — and we do not need to know which.

Comparing them naively would punish the accent rather than the errors, because Ghanaian English
legitimately realises θ as [t], æ as [a], and monophthongises FACE and GOAT. So both sides are
folded into coarse phone classes first, with all vowels collapsed to a single class. Consonant
manner/place is stable across accents; vowel quality is exactly what varies. What survives is a
measure of content agreement that is blind to accent.

OOV words are skipped rather than guessed, and clips whose text is mostly OOV are not scored on
agreement at all — a confident-looking number from 30% coverage would be worse than none.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ARPAbet -> coarse class. Vowels all collapse to V: their quality is precisely what a
# Ghanaian accent shifts, so scoring on it would measure accent, not correctness.
ARPA_CLASS = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "F": "f", "V": "v", "S": "s", "Z": "z", "SH": "S", "ZH": "S",
    "TH": "t", "DH": "d",          # Ghanaian English: think/this -> [t]/[d]
    "HH": "h", "CH": "C", "JH": "J",
    "M": "m", "N": "n", "NG": "N",
    "L": "l", "R": "r", "W": "w", "Y": "j",
    "ER": "V",  # non-rhotic: NURSE carries no consonantal r
}
for v in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OW", "OY", "UH", "UW"):
    ARPA_CLASS[v] = "V"

# Our 172-unit IPA inventory -> the same coarse classes.
IPA_CLASS = {
    "p": "p", "pʰ": "p", "b": "b", "t": "t", "tʰ": "t", "d": "d",
    "k": "k", "kʰ": "k", "ɡ": "g", "g": "g", "k͡p": "k", "ɡ͡b": "g", "g͡b": "g",
    "f": "f", "v": "v", "s": "s", "z": "z", "ʃ": "S", "ʒ": "S",
    "θ": "t", "ð": "d", "h": "h", "hʷ": "h", "ç": "h", "x": "h", "ɣ": "g",
    "t͡ʃ": "C", "tɕ": "C", "tʃ": "C", "d͡ʒ": "J", "dʒ": "J", "dz": "J", "dʑ": "J",
    "ts": "s", "tsʰ": "s",
    "m": "m", "n": "n", "ɲ": "n", "ŋ": "N", "ŋ͡m": "N", "m̩": "m", "n̩": "n",
    "l": "l", "r": "r", "ɾ": "r", "ɽ": "r", "ʁ": "r", "ɹ": "r",
    "w": "w", "j": "j", "ɥ": "j",
}
for v in ("a", "e", "i", "o", "u", "ɛ", "ɔ", "ə", "ɪ", "ʊ", "æ", "ʌ", "ɑ", "ɨ", "ʉ",
          "aː", "eː", "iː", "oː", "uː", "ɛː", "ɔː", "ɪː", "ʊː", "oɔ", "eɛ", "ie", "uo",
          "ia", "ua", "ai", "au", "ei", "ou", "ɛɔ"):
    IPA_CLASS[v] = "V"

WORD_RE = re.compile(r"[a-z']+")


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


def load_cmudict() -> dict[str, list[str]]:
    """First pronunciation per word.

    Prefers the standalone `cmudict` package over nltk: nltk ships an import guard that
    refuses to load `regex` when the working directory is on sys.path, which is exactly the
    situation when running a script from its own directory.
    """
    try:
        import cmudict as cmu
        return {w: prons[0] for w, prons in cmu.dict().items()}
    except ImportError:
        pass

    import nltk
    try:
        from nltk.corpus import cmudict
        d = cmudict.dict()
    except LookupError:
        nltk.download("cmudict", quiet=True)
        from nltk.corpus import cmudict
        d = cmudict.dict()
    return {w: prons[0] for w, prons in d.items()}


def ref_classes(text: str, cmu: dict) -> tuple[list[str], float]:
    """Expected coarse-class sequence from the text, plus dictionary coverage."""
    words = WORD_RE.findall((text or "").lower())
    if not words:
        return [], 0.0
    out, known = [], 0
    for w in words:
        pron = cmu.get(w)
        if pron is None:
            continue
        known += 1
        for ph in pron:
            cls = ARPA_CLASS.get(ph.rstrip("012"))
            if cls:
                out.append(cls)
    return out, known / len(words)


def hyp_classes(ipa: str) -> list[str]:
    out = []
    for u in (ipa or "").split(" "):
        cls = IPA_CLASS.get(u)
        if cls:
            out.append(cls)
    return out


def collapse(seq: list[str]) -> list[str]:
    """Drop repeats: gemination and vowel-length differences are not content errors."""
    out = []
    for c in seq:
        if not out or out[-1] != c:
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="out/eng_shards")
    ap.add_argument("--speakers", default="out/eng_speakers.parquet")
    ap.add_argument("--out", default="out/eng_selection.parquet")
    ap.add_argument("--target-hours", type=float, default=164.0,
                    help="how much English to keep; default matches the Twi side")
    ap.add_argument("--min-coverage", type=float, default=0.7,
                    help="minimum CMUdict word coverage to trust the agreement score")
    ap.add_argument("--max-uer", type=float, default=0.40,
                    help="reject above this coarse-class UER")
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--min-rate", type=float, default=10.0,
                    help="minimum phoneme rate in units/s. The default is tuned for a "
                         "second language the ASR was not trained on, where it "
                         "under-produces: this corpus's median is 7.7 units/s against 11.4 "
                         "for the in-domain language, so a low rate means dropped phonemes "
                         "rather than slow speech. Training on those would teach a TTS "
                         "model to swallow sounds.")
    ap.add_argument("--max-rate", type=float, default=25.0)
    ap.add_argument("--min-snr", type=float, default=15.0)
    ap.add_argument("--max-clip-frac", type=float, default=0.001)
    ap.add_argument("--max-silence-frac", type=float, default=0.5)
    ap.add_argument("--per-speaker-frac", type=float, default=0.08,
                    help="no speaker may exceed this share of the selection")
    args = ap.parse_args()

    cmu = load_cmudict()
    print(f"cmudict: {len(cmu)} words")

    t = ds.dataset(args.shards, format="parquet").to_table(
        columns=["id", "text", "ipa", "duration", "n_units"]).to_pydict()
    n = len(t["id"])
    print(f"{n} English clips, {sum(t['duration'])/3600:.0f} h")

    spk = pq.read_table(args.speakers).to_pydict()
    spk_of = dict(zip(spk["id"], spk["speaker"]))
    snr_of = dict(zip(spk["id"], spk.get("snr_est", [])))
    clip_of = dict(zip(spk["id"], spk.get("clip_frac", [])))
    sil_of = dict(zip(spk["id"], spk.get("silence_frac", [])))

    rows = []
    reasons: dict[str, int] = defaultdict(int)
    for i in range(n):
        cid = t["id"][i]
        dur = t["duration"][i]
        ref, cov = ref_classes(t["text"][i], cmu)
        hyp = hyp_classes(t["ipa"][i])
        uer = (edit_distance(collapse(ref), collapse(hyp)) / max(len(collapse(ref)), 1)
               if ref else 1.0)
        snr = snr_of.get(cid, 0.0)
        clipf = clip_of.get(cid, 0.0)
        silf = sil_of.get(cid, 0.0)
        rate = t["n_units"][i] / max(dur, 1e-3)

        why = []
        if not (args.min_dur <= dur <= args.max_dur):
            why.append("duration")
        if cov < args.min_coverage:
            why.append("low_coverage")
        elif uer > args.max_uer:
            why.append("text_audio_mismatch")
        if snr < args.min_snr:
            why.append("low_snr")
        if clipf > args.max_clip_frac:
            why.append("clipping")
        if silf > args.max_silence_frac:
            why.append("mostly_silence")
        if not (args.min_rate <= rate <= args.max_rate):
            why.append("bad_rate")
        for w in why:
            reasons[w] += 1

        rows.append({
            "id": cid, "speaker": spk_of.get(cid, "eng_unknown"), "duration": dur,
            "text": t["text"][i], "ipa": t["ipa"][i], "n_units": t["n_units"][i],
            "coverage": cov, "uer": uer, "snr_est": snr, "clip_frac": clipf,
            "silence_frac": silf, "eligible": not why,
        })

    elig = [r for r in rows if r["eligible"]]
    print(f"\neligible: {len(elig)}/{n} ({len(elig)/n:.1%}), "
          f"{sum(r['duration'] for r in elig)/3600:.0f} h")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  rejected {k:22} {v:7} ({v/n:5.1%})")

    # Rank by agreement first, then recording quality. Both are already thresholded, so this
    # is choosing the best of the acceptable rather than rescuing the unacceptable.
    elig.sort(key=lambda r: (r["uer"], -r["snr_est"]))

    budget = args.target_hours * 3600
    cap = budget * args.per_speaker_frac
    per: dict[str, float] = defaultdict(float)
    chosen, secs = [], 0.0
    for r in elig:
        if secs >= budget:
            break
        if per[r["speaker"]] + r["duration"] > cap:
            continue
        chosen.append(r)
        per[r["speaker"]] += r["duration"]
        secs += r["duration"]

    ids = {r["id"] for r in chosen}
    for r in rows:
        r["selected"] = r["id"] in ids

    print(f"\nselected {len(chosen)} clips, {secs/3600:.1f} h, "
          f"{len(per)} speakers (cap {args.per_speaker_frac:.0%} = {cap/3600:.1f} h each)")
    if chosen:
        u = np.array([r["uer"] for r in chosen])
        s = np.array([r["snr_est"] for r in chosen])
        d = np.array([r["duration"] for r in chosen])
        print(f"  agreement UER: median {np.median(u):.3f}, p95 {np.percentile(u,95):.3f}")
        print(f"  snr_est:       median {np.median(s):.1f} dB")
        print(f"  duration:      median {np.median(d):.1f} s, mean {d.mean():.1f} s")

    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), args.out,
                   compression="zstd")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
