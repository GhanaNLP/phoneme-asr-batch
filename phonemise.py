"""Batch IPA phonemisation of a speech dataset with ghananlpcommunity/ghana-speech-phoneme-asr.

The model card shows sherpa-onnx, which is CPU-only and decodes one clip at a time. That is
right for a demo and wrong for a corpus: 172 hours would take days. This runs the fairseq2
checkpoint on a GPU in batches and does ~700x realtime on one H200, so the same 172 hours
takes about a quarter of an hour.

Three details are easy to get wrong and all of them produce plausible-looking IPA rather
than an error, so they are handled explicitly here:

  normalisation  the recipe sets `normalize_audio: true`, so the encoder expects
                 per-utterance layer-normalised waveforms. Feeding raw audio costs ~12% UER.
                 Statistics come from the valid samples only — never the padding.
  head slicing   the CTC head is 9812 wide because `arch: 300m` carries omniASR's own
                 vocabulary size. Only the first 176 ids were ever training targets; the
                 rest are dead and get sliced off, matching tokens.txt.
  layer drop     the training config leaves layer_drop_p at 0.1. eval() does not disable it
                 in every fairseq2 version, so it is zeroed by hand.

Output is a sidecar table keyed by the source clip filename, with no audio in it, so it
joins onto the source dataset instead of republishing the waveforms.
"""
from __future__ import annotations

import argparse
import io
import json
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio
import yaml

MODEL_SR = 16000  # what the omniASR encoder was trained on
BLANK = 0  # <pad>, id 0 in tokens.txt
DOWNSAMPLE = 320  # feature extractor stride: samples per output frame
HF_MODEL = "ghananlpcommunity/ghana-speech-phoneme-asr"


# ---------------------------------------------------------------- model

def resolve_model(spec: str) -> Path:
    """A local directory, or an HF repo id to pull the checkpoint and tokens from."""
    p = Path(spec)
    if p.exists():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(spec, allow_patterns=["checkpoint/*", "config/*", "onnx/tokens.txt"]))


def read_tokens(path: Path) -> list[str]:
    """tokens.txt holds '<symbol> <id>' per line, already in real IPA."""
    toks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        sym, idx = line.rsplit(" ", 1)
        toks[int(idx)] = sym
    return [toks[i] for i in range(len(toks))]


def load_model(root: Path, device: str = "cuda"):
    from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
    from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrConfig, create_wav2vec2_asr_model

    vocab = read_tokens(root / "onnx/tokens.txt")
    mc = yaml.safe_load((root / "config/model_arch.yaml").read_text())["model_config"]
    enc = Wav2Vec2EncoderConfig(**mc["encoder_config"])
    cfg = Wav2Vec2AsrConfig(
        **{**{k: v for k, v in mc.items() if k != "encoder_config"}, "encoder_config": enc}
    )

    sd = torch.load(root / "checkpoint/model/pp_00/tp_00/sdp_00.pt", map_location="cpu",
                    weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    model = create_wav2vec2_asr_model(cfg)
    model.load_state_dict(sd)
    model.eval()
    for m in model.modules():  # regulariser, must be off for inference
        if hasattr(m, "layer_drop_p"):
            m.layer_drop_p = 0.0

    return model.to(device), len(vocab), vocab


# ---------------------------------------------------------------- audio in

def decode_shard(path: Path, pool: ThreadPoolExecutor, audio_col: str,
                 text_col: str | None) -> dict:
    """Read a parquet shard and decode every embedded clip to float32 samples.

    soundfile releases the GIL, so a thread pool genuinely parallelises this and the
    decoded arrays never cross a process boundary.
    """
    cols = [audio_col] + ([text_col] if text_col else [])
    t = pq.read_table(path, columns=cols).to_pydict()
    audio = t[audio_col]

    def raw(a):
        # HF Audio columns are a {bytes, path} struct; some datasets store bare wav bytes.
        return a["bytes"] if isinstance(a, dict) else a

    def one(a):
        w, sr = sf.read(io.BytesIO(raw(a)), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        return w, int(sr)

    decoded = list(pool.map(one, audio))

    # Without a path we have no natural key, so synthesise a stable one from provenance:
    # the shard stem plus the row index is reproducible and unique across the corpus.
    stem = path.stem
    paths, ids = [], []
    for i, a in enumerate(audio):
        p = a.get("path") if isinstance(a, dict) else None
        paths.append(p)
        ids.append(Path(p).stem if p else f"{stem}_{i:06d}")

    return {
        "shard": path.name,
        "ids": ids,
        "paths": paths,
        "texts": t[text_col] if text_col else [None] * len(audio),
        "waves": [w for w, _ in decoded],
        "srs": [sr for _, sr in decoded],
    }


def prefetch(shards: list[Path], depth: int, **kw):
    """Yield decoded shards while `depth` more decode ahead, so CPU overlaps GPU."""
    q: queue.Queue = queue.Queue(maxsize=depth)

    def worker():
        try:
            for s in shards:
                q.put(decode_shard(s, **kw))
        except Exception as e:
            q.put(e)
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item


def make_batches(lengths: list[int], idx: list[int], budget: int,
                 max_rows: int) -> list[list[int]]:
    """Length-sorted batches capped by *padded* sample count rather than row count.

    A fixed batch size either wastes the GPU on short clips or runs out of memory on long
    ones. Budgeting padded samples means short clips ride hundreds at a time and 30-second
    clips fall back to a handful, with padding waste small either way.
    """
    order = sorted(idx, key=lambda i: lengths[i])
    batches, cur = [], []
    for i in order:
        longest = lengths[i]  # ascending, so this is the max of the growing batch
        if cur and ((len(cur) + 1) * longest > budget or len(cur) >= max_rows):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


# ---------------------------------------------------------------- gpu

@torch.inference_mode()
def run_batch(model, keep: int, waves: list[np.ndarray], srs: list[int],
              cache: dict, device: str = "cuda") -> list[list[int]]:
    """Decode one batch of same-sample-rate clips to CTC id sequences."""
    from fairseq2.nn.batch_layout import BatchLayout

    sr = srs[0]
    assert all(s == sr for s in srs), "batch must be single-sample-rate"

    pad = max(len(w) for w in waves)
    x = torch.zeros(len(waves), pad, dtype=torch.float32, device=device)
    for i, w in enumerate(waves):
        x[i, : len(w)] = torch.from_numpy(w).to(device, non_blocking=True)

    if sr != MODEL_SR:
        rs = cache.get(sr)
        if rs is None:
            # Same resampler family the training ingest used (torchaudio sinc/Hann).
            rs = cache[sr] = torchaudio.transforms.Resample(
                sr, MODEL_SR, dtype=torch.float32).to(device)
        x = rs(x)
        lens = [min(max(1, round(len(w) * MODEL_SR / sr)), x.shape[1]) for w in waves]
    else:
        lens = [len(w) for w in waves]

    # normalize_audio: true — per-utterance layer norm over valid samples only.
    for i, n in enumerate(lens):
        seg = x[i, :n]
        x[i, :n] = (seg - seg.mean()) / torch.sqrt(seg.var(unbiased=False) + 1e-5)
        x[i, n:] = 0.0

    bl = BatchLayout(shape=(x.shape[0], x.shape[1]), seq_lens=lens)
    with torch.autocast(device, dtype=torch.bfloat16):
        logits, out_bl = model(x, bl)

    ids = logits[..., :keep].float().argmax(-1).cpu().numpy()
    out_lens = getattr(out_bl, "seq_lens", None) or [max(1, n // DOWNSAMPLE) for n in lens]
    return [ids[i, : int(out_lens[i])].tolist() for i in range(len(waves))]


def ctc_collapse(ids: list[int], vocab: list[str]) -> list[str]:
    """Greedy CTC: drop repeats, then blanks. Units stay whole — never split k͡p by char."""
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(vocab[i])
        prev = i
    return out


def phonemise_shard(sh: dict, model, keep, vocab, budget, max_rows, device) -> list[list[str]]:
    n = len(sh["waves"])
    lens16 = [round(len(w) * MODEL_SR / sr) for w, sr in zip(sh["waves"], sh["srs"])]

    by_sr: dict[int, list[int]] = defaultdict(list)
    for i, sr in enumerate(sh["srs"]):
        by_sr[sr].append(i)

    units: list[list[str]] = [None] * n  # type: ignore[list-item]
    cache: dict = {}
    for sr, idx in by_sr.items():
        for b in make_batches(lens16, idx, budget, max_rows):
            out = run_batch(model, keep, [sh["waves"][i] for i in b],
                            [sr] * len(b), cache, device)
            for i, o in zip(b, out):
                units[i] = ctc_collapse(o, vocab)
    return units


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=HF_MODEL,
                    help="local checkpoint dir, or an HF repo id")
    ap.add_argument("--data", required=True,
                    help="directory of parquet shards holding the audio")
    ap.add_argument("--out", default="out/shards", help="one parquet of results per shard")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="text",
                    help="carried through for reference; pass '' if absent")
    ap.add_argument("--budget", type=int, default=MODEL_SR * 900,
                    help="padded samples per batch at 16 kHz (default 900 s)")
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--threads", type=int, default=16, help="audio decode threads")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="substring filter on shard filename")
    ap.add_argument("--limit-shards", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    root = resolve_model(args.model)
    model, keep, vocab = load_model(root, args.device)
    print(f"model {args.model}: head sliced to {keep} tokens", flush=True)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    shards = sorted(Path(args.data).glob("*.parquet"))
    if args.only:
        shards = [s for s in shards if args.only in s.name]
    todo = [s for s in shards if not (outdir / f"{s.stem}.parquet").exists()]
    if args.limit_shards:
        todo = todo[: args.limit_shards]
    print(f"{len(todo)} of {len(shards)} shards to do", flush=True)
    if not todo:
        return

    pool = ThreadPoolExecutor(max_workers=args.threads)
    t_start = time.time()
    rows = secs = 0

    for sh in prefetch(todo, depth=1, pool=pool, audio_col=args.audio_col,
                       text_col=args.text_col or None):
        t0 = time.time()
        units = phonemise_shard(sh, model, keep, vocab, args.budget, args.max_rows,
                                args.device)
        n = len(units)
        dur = [len(w) / sr for w, sr in zip(sh["waves"], sh["srs"])]
        audio_secs = sum(dur)

        pq.write_table(
            pa.table({
                "id": sh["ids"],
                "audio_path": sh["paths"],
                "shard": pa.array([sh["shard"]] * n, pa.string()),
                "row": pa.array(list(range(n)), pa.int32()),
                "duration": pa.array(dur, pa.float32()),
                "text": sh["texts"],
                "ipa": [" ".join(u) for u in units],
                "ipa_units": units,
                "n_units": pa.array([len(u) for u in units], pa.int32()),
            }),
            outdir / f"{Path(sh['shard']).stem}.parquet", compression="zstd",
        )

        rows += n
        secs += audio_secs
        dt = time.time() - t0
        print(f"{sh['shard']:34} {n:6} clips {audio_secs/60:7.1f} min audio {dt:6.1f}s "
              f"{audio_secs/dt:6.0f}x RT | total {rows} clips, {secs/3600:.1f} h in "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)

    print(f"\ndone: {rows} clips, {secs/3600:.2f} h audio, "
          f"{(time.time()-t_start)/60:.1f} min wall", flush=True)


if __name__ == "__main__":
    main()
