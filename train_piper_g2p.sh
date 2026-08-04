#!/bin/bash
# Finetune Piper on the bilingual Twi + Ghanaian English IPA corpus.
#
# Convergence policy, and why it is not a plain early-stop:
#
# Piper's own maintainers note that val_mel (mel L1) saturates early in VITS while the
# adversarial losses keep removing audible artifacts, so an EarlyStopping on it fires well
# before the audio is clean. So we train open-ended and select a checkpoint instead of
# stopping on a loss:
#
#   val_mel   top-5 kept  - reconstruction fidelity
#   val_mos   top-5 kept  - UTMOS22 no-reference perceptual quality (naturalness)
#   round-trip phoneme UER, measured separately by tts_eval.py - intelligibility
#
# Note: --data.espeak_voice is required by the CLI even in phoneme_ids mode, where the ids
# come from the CSV and espeak is never called. The value is inert. --data.phoneme_type text
# is set for the same reason: it selects a phonemizer that is never used, and avoids
# constructing the espeak one, whose compiled bridge may not be built.
#
# validation_split is small on purpose. The default validates on thousands of utterances,
# which at UTMOS speed costs ~9 minutes every 2000 steps — around 20% of total throughput
# for a metric that is a mean and stabilises in a few hundred samples.
#
# The three cover different failure modes. val_mos cannot tell whether the right phonemes
# were said; the round-trip UER cannot tell whether the result sounds pleasant. Training
# stops when val_mos and round-trip UER have both plateaued, which is judged from the logs
# rather than asserted up front.
set -euo pipefail

ROOT=/mnt/volume_d2wey28/projects/tts-twi
DATA=$ROOT/data22k
PY=$ROOT/.venv-piper/bin/python
RUN=${RUN:-$ROOT/runs/piper-bilingual}
PRETRAINED="$ROOT/pretrained/en/en_US/libritts_r/medium/epoch=404-step=1887300.ckpt"
ADAPTED=$ROOT/pretrained/libritts_r_medium.adapted.ckpt
BATCH=${BATCH:-32}

mkdir -p "$RUN"

# metadata_train_g2p.csv is pipe-delimited: utt|speaker|text|phoneme_ids
NSPK=$(cut -d'|' -f2 "$DATA/metadata_train_g2p.csv" | sort -u | grep -c .)
if [ "$NSPK" -lt 2 ]; then
  echo "refusing to train: found only $NSPK speaker(s) in $DATA/metadata_train_g2p.csv" >&2
  exit 1
fi
echo "speakers: $NSPK"

if [ ! -f "$ADAPTED" ]; then
  echo "=== adapting pretrained checkpoint to $NSPK speakers ==="
  $PY /mnt/volume_d2wey28/projects/phonemise-new-twi-tts/adapt_piper_ckpt.py \
      --ckpt "$PRETRAINED" --out "$ADAPTED" --num-speakers "$NSPK"
fi

echo "=== training ==="
exec $PY -m piper.train fit \
  --data.voice_name gh_bilingual_ipa \
  --data.csv_path "$DATA/metadata_train_g2p.csv" \
  --data.audio_dir "$DATA/wav" \
  --data.dataset_type phoneme_ids \
  --data.phonemes_path "$DATA/phonemes_g2p.json" \
  --data.espeak_voice en-us \
  --data.phoneme_type text \
  --data.num_symbols 256 \
  --data.cache_dir "$RUN/cache" \
  --data.config_path "$RUN/config.json" \
  --data.batch_size "$BATCH" \
  --data.num_workers 12 \
  --data.validation_split 0.005 \
  --model.num_speakers "$NSPK" \
  --model.sample_rate 22050 \
  --model.warmstart_ckpt "$ROOT/pretrained/continue_g2p.ckpt" \
  --model.mos_metric utmos \
  --trainer.default_root_dir "$RUN" \
  --trainer.max_epochs -1 \
  --trainer.precision bf16-mixed \
  --trainer.val_check_interval 2000 \
  --trainer.log_every_n_steps 50 \
  --trainer.accelerator gpu \
  --trainer.devices 1
