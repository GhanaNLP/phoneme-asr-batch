"""Turn arbitrary text into a synthesis manifest, so the models can speak unseen sentences.

Twi goes through GhanaNLP's ghana-g2p, which is the right tool and — as it turns out — almost
certainly the source of the ASR's own 172-unit inventory: every unit it emits for Asante Twi is
already in the trained phoneme map, and its style matches the training targets exactly
(aspirated pʰ/tʰ/kʰ, ɾ for r, ɪ in closed syllables).

English is not covered by ghana-g2p, so it goes through CMUdict with an ARPAbet mapping folded
to Ghanaian English. The fold is not guesswork: it was read off the unit frequencies in the
English half of the training data, and the data disagrees with the obvious choices in ways that
matter.

    k  3.14%   kʰ 0.00%      the English half uses PLAIN stops, where Twi is aspirated
    t  6.66%   tʰ 0.02%
    r  2.30%   ɾ  0.06%      plain r, not the Twi tap
    d  5.79%   ð  0.00%      "th" is realised as [d]
    i  8.08%   ɪ  1.71%      i preferred over ɪ

Using Twi's aspirated stops for English would put the input out of the distribution the model
learned, so the two languages get different consonant sets and the language token reinforces
the split. Unstressed vowels map to full vowels rather than schwa, which is both a documented
feature of Ghanaian English and what the data shows (ə does not reach the top 30 units).
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

FIELDS = ["id", "speaker", "split", "duration", "text", "ipa", "n_units", "language"]
WORD_RE = re.compile(r"[A-Za-z']+")

# ARPAbet -> our inventory, Ghanaian English. Plain stops, non-rhotic, monophthongal
# FACE/GOAT, no schwa reduction. See the module docstring for the evidence.
ARPA_IPA: dict[str, list[str]] = {
    "AA": ["a"], "AE": ["a"], "AH": ["a"], "AO": ["ɔ"],
    "AW": ["a", "u"], "AY": ["a", "i"], "EH": ["ɛ"], "ER": ["ɛ"],
    "EY": ["e"], "IH": ["ɪ"], "IY": ["i"], "OW": ["o"],
    "OY": ["ɔ", "i"], "UH": ["ʊ"], "UW": ["u"],
    "B": ["b"], "CH": ["t͡ʃ"], "D": ["d"], "DH": ["d"], "F": ["f"], "G": ["ɡ"],
    "HH": ["h"], "JH": ["d͡ʒ"], "K": ["k"], "L": ["l"], "M": ["m"], "N": ["n"],
    "NG": ["ŋ"], "P": ["p"], "R": ["r"], "S": ["s"], "SH": ["ʃ"], "T": ["t"],
    "TH": ["t"], "V": ["v"], "W": ["w"], "Y": ["j"], "Z": ["z"], "ZH": ["ʒ"],
}


def twi_g2p(text: str, dialect: str):
    from ghana_g2p import GhanaG2P
    g = GhanaG2P(dialect)
    units = [u for u in g.ipa(text, sep=" ", punctuation=True).split(" ") if u]
    return units


def english_g2p(text: str, cmu: dict) -> tuple[list[str], list[str]]:
    units: list[str] = []
    oov: list[str] = []
    for tok in re.findall(r"[A-Za-z']+|[.,?!;:]", text):
        if tok in ".,?!;:":
            units.append(tok)
            continue
        pron = cmu.get(tok.lower())
        if pron is None:
            oov.append(tok)
            continue
        for ph in pron:
            m = ARPA_IPA.get(ph.rstrip("012"))
            if m:
                units.extend(m)
    return units, oov


SPAN_RE = re.compile(r"\[([^\]]*)\]")


def mixed_g2p(text: str, dialect: str, cmu: dict) -> tuple[list[str], list[str]]:
    """Code-switched line: [bracketed] spans are English, everything else is Twi.

    Each span is phonemised by its own language's rules, which matters because the two
    halves of the training data use different sub-inventories — Twi is aspirated (kʰ, pʰ,
    tʰ, ɾ) and the English half is not (k, p, t, r). Phonemising a switched sentence wholly
    as one language would put half of it out of distribution.

    Nothing in training was code-switched, so this is an extrapolation. It is representable
    though: both languages draw on one shared inventory, and the model consumes phonemes
    rather than spelling, so it never has to decide which language a word belongs to.
    """
    units: list[str] = []
    oov: list[str] = []
    pos = 0
    for m in SPAN_RE.finditer(text):
        twi_part = text[pos:m.start()].strip()
        if twi_part:
            units += twi_g2p(twi_part, dialect)
        eng_part = m.group(1).strip()
        if eng_part:
            u, o = english_g2p(eng_part, cmu)
            units += u
            oov += o
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        units += twi_g2p(tail, dialect)
    return units, oov


def pick_speakers(manifest: Path, language: str, n: int) -> list[str]:
    """Busiest speakers for a language: most training data, so the best-learned voices."""
    counts: Counter = Counter()
    with open(manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                                escapechar="\\"):
            if r.get("language") == language:
                counts[r["speaker"]] += 1
    return [s for s, _ in counts.most_common(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="manifest TSV to write")
    ap.add_argument("--train-manifest", default=None,
                    help="training manifest, used to pick well-trained speakers")
    ap.add_argument("--twi-file", default=None, help="one Twi sentence per line")
    ap.add_argument("--eng-file", default=None, help="one English sentence per line")
    ap.add_argument("--mixed-file", default=None,
                    help="code-switched lines; [bracketed] spans are English, rest is Twi")
    ap.add_argument("--mixed-speakers", default=None,
                    help="comma-separated speakers for mixed lines; defaults to both "
                         "languages' busiest voices, so the switch can be heard from each side")
    ap.add_argument("--dialect", default="Asante Twi")
    ap.add_argument("--speakers-per-language", type=int, default=2)
    ap.add_argument("--twi-speaker", default=None)
    ap.add_argument("--eng-speaker", default=None)
    ap.add_argument("--id-map", default=None,
                    help="config.json; warns about units the model cannot say")
    args = ap.parse_args()

    known = None
    if args.id_map:
        import json
        known = set(json.loads(Path(args.id_map).read_text())["phoneme_id_map"])

    jobs: list[tuple[str, list[str]]] = []
    if args.twi_file:
        jobs.append(("twi", [l.strip() for l in
                             Path(args.twi_file).read_text(encoding="utf-8").splitlines()
                             if l.strip()]))
    if args.eng_file:
        jobs.append(("eng", [l.strip() for l in
                             Path(args.eng_file).read_text(encoding="utf-8").splitlines()
                             if l.strip()]))
    if args.mixed_file:
        jobs.append(("mixed", [l.strip() for l in
                               Path(args.mixed_file).read_text(encoding="utf-8").splitlines()
                               if l.strip()]))
    if not jobs:
        raise SystemExit("pass --twi-file, --eng-file and/or --mixed-file")

    cmu = None
    rows = []
    for lang, sents in jobs:
        if lang == "mixed":
            if args.mixed_speakers:
                speakers = [s.strip() for s in args.mixed_speakers.split(",") if s.strip()]
            else:
                # One voice from each language: a Twi voice has never uttered English and
                # vice versa, so which side the speaker comes from is itself a variable.
                speakers = (pick_speakers(Path(args.train_manifest), "twi", 1)
                            + pick_speakers(Path(args.train_manifest), "eng", 1))
        else:
            forced = args.twi_speaker if lang == "twi" else args.eng_speaker
            if forced:
                speakers = [forced]
            elif args.train_manifest:
                speakers = pick_speakers(Path(args.train_manifest), lang,
                                         args.speakers_per_language)
            else:
                raise SystemExit("pass --train-manifest or an explicit speaker")
        print(f"{lang}: {len(sents)} sentences x {len(speakers)} speakers {speakers}")

        for si, text in enumerate(sents):
            if lang == "twi":
                units = twi_g2p(text, args.dialect)
                oov: list[str] = []
            else:
                if cmu is None:
                    import cmudict
                    cmu = {w: p[0] for w, p in cmudict.dict().items()}
                if lang == "mixed":
                    units, oov = mixed_g2p(text, args.dialect, cmu)
                else:
                    units, oov = english_g2p(text, cmu)
            if oov:
                print(f"  [oov] {lang} #{si}: {' '.join(oov)}")
            if known is not None:
                bad = sorted({u for u in units if u not in known})
                if bad:
                    print(f"  [unsayable] {lang} #{si}: {bad}")
                    units = [u for u in units if u in known]
            if not units:
                print(f"  [skip] {lang} #{si}: no units")
                continue

            # The language token is per-utterance, so a switched sentence has to pick one.
            # Twi is the matrix language in Ghanaian code-switching — English words are
            # inserted into a Twi frame — so the frame's token is the honest choice.
            token_lang = "twi" if lang == "mixed" else lang
            for spk in speakers:
                rows.append({
                    "id": f"ood_{lang}_{si:02d}_{spk}", "speaker": spk, "split": "test",
                    "duration": 0.0, "text": text.replace("\t", " "),
                    "ipa": " ".join(units), "n_units": len(units),
                    "language": token_lang,
                })
            print(f"  {lang} #{si}: {len(units)} units | {' '.join(units[:24])}"
                  f"{' …' if len(units) > 24 else ''}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t",
                           quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} utterances to {args.out}")


if __name__ == "__main__":
    main()
