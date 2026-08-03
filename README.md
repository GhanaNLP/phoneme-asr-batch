# Batch IPA phonemisation with `ghana-speech-phoneme-asr`

GPU batch inference for [`ghananlpcommunity/ghana-speech-phoneme-asr`](https://huggingface.co/ghananlpcommunity/ghana-speech-phoneme-asr)
— a CTC model that turns speech in **42 Ghanaian and West African languages** into IPA
phonemes. Point it at a corpus, get one IPA transcription per clip.

The model card's example uses `sherpa-onnx`, which is CPU-only and decodes a single clip per
call. That is the right way to try the model and the wrong way to run a corpus through it.
This repo runs the fairseq2 checkpoint on a GPU in batches:

| | model card (`sherpa-onnx`, int8, CPU) | this repo (fairseq2, bf16, 1×H200) |
|---|---|---|
| throughput | ~5–10× realtime | **~780× realtime** |
| 172 h corpus | ~1 day | **~13 minutes** |

Built to phonemise [`ghanaopendata/new-twi-tts-aligned`](https://huggingface.co/datasets/ghanaopendata/new-twi-tts-aligned)
(161,398 clips, 172 h) → [`ghanaopendata/new-twi-tts-aligned-ipa`](https://huggingface.co/datasets/ghanaopendata/new-twi-tts-aligned-ipa).
It is not specific to that dataset: any directory of parquet shards with an `audio` column
works, at any sample rate.

## Why not just use the ONNX model?

Because three things about this model are easy to get wrong, and **every one of them produces
plausible-looking IPA instead of an error.** If you write your own inference loop, check all
three. They are the reason this repo exists.

### 1. The encoder expects normalised waveforms

The training recipe sets `normalize_audio: true`. The encoder was fed per-utterance
layer-normalised audio — zero mean, unit variance. Feed it raw waveforms and it still emits
fluent-looking IPA, just **12% worse**:

```
raw waveforms       12.2% UER vs the reference decoder
normalised           1.7% UER vs the reference decoder   (the rest is int8 quantisation)
```

Statistics must come from the **valid samples only**. Include the zero padding and a short
clip in a long batch gets scaled differently than the same clip alone.

### 2. The CTC head is 9812 wide, but only 176 ids are real

`arch: 300m` carries omniASR's own vocabulary size, and the recipe does not derive it from
the tokenizer. Ids 176+ were never training targets. `argmax` over the full head can land on
a dead id; slice to `len(tokens.txt)` first.

### 3. Layer drop is still on

The recipe trains with `layer_drop_p: 0.1`. Depending on your fairseq2 version, `eval()`
does not necessarily disable it — and if it is live at inference, your output is
nondeterministic. This repo zeroes it explicitly.

There is also a fourth, milder one: wav2vec2's convolutional front-end is not
padding-masked, so batching shifts results slightly versus one-clip-at-a-time
(~0.2% UER with length-sorted batches). That is far below the model's own error rate, but it
means batched and unbatched output are not bit-identical, and you should not be alarmed when
they differ.

## Install

Needs a CUDA GPU, `fairseq2` (which brings the model architecture), and `torchaudio`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`fairseq2` ships native extensions; if the wheel does not resolve for your CUDA/Python
combination, follow the [fairseq2 install matrix](https://facebookresearch.github.io/fairseq2/stable/getting_started/installation/).

## Use

Get a corpus as parquet shards with an embedded `audio` column (this is what
`huggingface-cli download` gives you for any audio dataset on the Hub):

```bash
huggingface-cli download ghanaopendata/new-twi-tts-aligned \
    --repo-type dataset --local-dir data/raw --max-workers 16
```

**Always validate first** — it takes under a minute and catches a wrong front-end before you
spend a GPU-hour producing confident garbage:

```bash
python validate.py --shard data/raw/data/train-00000-of-00054.parquet --n 48
```

```
production batching vs single             45/48 exact, UER 0.1739%
tight batching (8 rows) vs single         44/48 exact, UER 0.2103%
one unsorted batch vs single              44/48 exact, UER 0.2319%
sherpa int8 (reference) vs torch          28/48 exact, UER 1.7341%
```

The number that matters is the last one. **If `sherpa vs torch` is above a few percent, your
audio front-end is wrong** — stop and fix it. The batching rows should be a fraction of a
percent.

Then run the corpus:

```bash
python phonemise.py --data data/raw/data --out out/shards
```

One parquet of results per input shard, and finished shards are skipped, so it is safe to
interrupt and re-run. Merge into a publishable sidecar dataset:

```bash
python build_dataset.py --shards out/shards --out out/dataset \
    --repo your-org/your-dataset-ipa --push
```

### Other datasets

```bash
python phonemise.py \
    --data data/raw/data \
    --audio-col audio \
    --text-col transcript \   # carried through for reference; use '' if absent
    --budget 14400000 \       # padded samples per batch at 16 kHz (default = 900 s)
    --max-rows 256
```

Sample rate is read per clip and resampled to 16 kHz on the GPU, so mixed-rate corpora are
fine. Clips must be **under 40 seconds** — that is a model constraint. Segment longer audio
first.

### Tuning throughput

`--budget` is the real knob: it caps *padded* samples per batch rather than row count, so
short clips ride hundreds at a time while 30-second clips fall back to a handful. Raise it
until you are near VRAM capacity. `--max-rows` is a safety cap for corpora of very short
clips. `--threads` sets audio-decode parallelism; decode runs one shard ahead of the GPU on
a prefetch thread, so if you see the GPU idling between shards, raise it.

## Output

`phonemise.py` writes a **sidecar** table — no audio, keyed by the source clip filename, so
it joins back onto the source dataset without republishing waveforms.

| column | meaning |
|---|---|
| `id` | clip id, e.g. `segment_042811` — the join key |
| `audio_path` | source `audio.path` |
| `shard`, `row` | provenance back to the exact source row |
| `duration` | seconds |
| `text` | source text, carried through unchanged |
| `ipa` | phonemes, **space-separated** |
| `ipa_units` | the same phonemes as a list |
| `n_units` | unit count |

### Never split IPA by character

Units are multi-character — `kʰ`, `t͡ʃ`, `k͡p`, `hʷ`, `iː`. The inventory is 172 units, not
172 characters.

```python
units = row["ipa"].split(" ")   # correct
units = list(row["ipa"])        # wrong — tears k͡p into three characters
```

## Interpreting the phonemes

- **They are predictions, not ground truth.** The model scores 17.1% phoneme UER on Asante
  Twi, 12.4% on Fante, and varies widely across its 42 languages — check
  `eval_results.json` on the model card for yours.
- **Punctuation is guessed.** The model emits punctuation, but punctuation has no acoustic
  realisation, so its error rate there is ~30% even where phonemes are good. Filter it out
  against a known punctuation set if you do not want it.
- **They follow the speech, not the spelling.** Where a speaker elides or reduces, the IPA
  reflects that and will not match a rule-based grapheme-to-phoneme rendering of the text.
  That is the reason to use an acoustic model — and the reason `ipa` and `text` will
  legitimately disagree.

## Files

| file | what it does |
|---|---|
| `phonemise.py` | batch GPU inference over parquet shards |
| `validate.py` | front-end and batching checks against the reference decoder |
| `build_dataset.py` | merge shards, write the dataset card, push to the Hub |

## License

Code: MIT. The model and any dataset you produce carry their own licenses — the Twi sidecar
above is `cc-by-nc-4.0`, inherited from the source audio.
