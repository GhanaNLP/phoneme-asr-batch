"""Derive pseudo-speaker labels for a corpus that has none.

new-twi-tts-aligned is Ghanaian news broadcast audio: hundreds of voices, no speaker column.
Most light TTS architectures (VITS, Piper, MeloTTS, Matcha, Glow-TTS, FastSpeech2) condition on
a speaker id, and training them single-speaker on many voices yields an averaged, unstable
timbre. So we manufacture the labels.

Two stages:

  embed    ECAPA-TDNN x-vectors, GPU-batched, one .npz per source shard so it resumes.
  cluster  161k x 161k is 100 GB of pairwise distance, so agglomerative clustering cannot be
           run directly. Instead over-segment with k-means into many more centroids than
           there are speakers, then agglomerate the *centroids* at a cosine threshold and
           propagate labels back. This is the standard trick for diarising at corpus scale
           and it keeps the threshold interpretable: 0.7 cosine is roughly ECAPA's
           same-speaker operating point.

The output is deliberately conservative. Over-splitting one real speaker into two pseudo-ids
costs a TTS model very little — it just learns two nearly identical embeddings. Merging two
real speakers into one id is what produces a muddy voice, so the threshold errs high.
"""
from __future__ import annotations

import argparse
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

ECAPA_SR = 16000
MAX_SECS = 8.0  # speaker identity saturates well before this; caps padding cost


def acoustic_metrics(w: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Cheap per-clip recording-quality signals, computed while the audio is already in RAM.

    snr_est compares a high energy percentile against a low one, which approximates
    speech-vs-noise-floor without needing a VAD. clip_frac catches recordings driven into
    the rails. silence_frac catches clips that are mostly dead air.
    """
    n = max(int(0.02 * sr), 1)
    trim = len(w) - (len(w) % n)
    if trim < n:
        return 0.0, 0.0, 1.0
    e = (w[:trim].reshape(-1, n) ** 2).mean(axis=1) + 1e-12
    hi, lo = np.percentile(e, 90), np.percentile(e, 10)
    snr = 10.0 * float(np.log10(hi / lo))
    clip_frac = float((np.abs(w) > 0.99).mean())
    silence_frac = float((e < lo * 2.0).mean())
    return snr, clip_frac, silence_frac


def embed_shards(src: Path, outdir: Path, batch: int, threads: int, device: str,
                 audio_col: str = "audio") -> None:
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    outdir.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.parquet"))
    todo = [s for s in shards if not (outdir / f"{s.stem}.npz").exists()]
    print(f"embedding {len(todo)} of {len(shards)} shards", flush=True)
    if not todo:
        return

    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(outdir.parent / "ecapa"), run_opts={"device": device})
    pool = ThreadPoolExecutor(max_workers=threads)
    rs_cache: dict = {}

    for sh in todo:
        t = pq.read_table(sh, columns=[audio_col]).to_pydict()[audio_col]

        def one(a):
            raw = a["bytes"] if isinstance(a, dict) else a
            w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            return w, int(sr)

        decoded = list(pool.map(one, t))
        # Mirrors phonemise.py: fall back to shard stem + row index when there is no path.
        ids = [Path(a["path"]).stem if isinstance(a, dict) and a.get("path")
               else f"{sh.stem}_{i:06d}" for i, a in enumerate(t)]
        metrics = np.array([acoustic_metrics(w, sr) for w, sr in decoded],
                           dtype=np.float32)

        # Sort by length so padding inside a batch stays small.
        order = sorted(range(len(decoded)), key=lambda i: len(decoded[i][0]))
        embs = np.zeros((len(decoded), 192), dtype=np.float32)

        for i in range(0, len(order), batch):
            grp = order[i: i + batch]
            sr = decoded[grp[0]][1]
            assert all(decoded[j][1] == sr for j in grp), "mixed sample rates in batch"
            cap = int(MAX_SECS * sr)
            wavs = [decoded[j][0][:cap] for j in grp]
            pad = max(len(w) for w in wavs)
            x = torch.zeros(len(wavs), pad, dtype=torch.float32, device=device)
            for k, w in enumerate(wavs):
                x[k, : len(w)] = torch.from_numpy(w).to(device)

            if sr != ECAPA_SR:
                rs = rs_cache.get(sr)
                if rs is None:
                    rs = rs_cache[sr] = torchaudio.transforms.Resample(
                        sr, ECAPA_SR).to(device)
                x = rs(x)
                lens = [min(round(len(w) * ECAPA_SR / sr), x.shape[1]) for w in wavs]
            else:
                lens = [len(w) for w in wavs]

            # wav_lens is relative, and it is what masks the padding out of the pooling.
            rel = torch.tensor([n / x.shape[1] for n in lens], device=device)
            with torch.inference_mode():
                e = enc.encode_batch(x, wav_lens=rel).squeeze(1).float().cpu().numpy()
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
            embs[grp] = e

        np.savez(outdir / f"{sh.stem}.npz", ids=np.array(ids), emb=embs, metrics=metrics,
                 metric_names=np.array(["snr_est", "clip_frac", "silence_frac"]))
        print(f"  {sh.name}: {len(ids)} clips", flush=True)


def cluster(embdir: Path, out: Path, threshold: float, centroids: int, seed: int,
            prefix: str = "spk") -> None:
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans

    files = sorted(embdir.glob("*.npz"))
    ids = np.concatenate([np.load(f)["ids"] for f in files])
    E = np.concatenate([np.load(f)["emb"] for f in files]).astype(np.float32)
    mets = [np.load(f) for f in files]
    M = (np.concatenate([m["metrics"] for m in mets]).astype(np.float32)
         if "metrics" in mets[0] else None)
    print(f"{len(ids)} embeddings", flush=True)

    k = min(centroids, len(E) // 4)
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=5,
                         max_iter=200).fit(E)
    C = km.cluster_centers_
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
    print(f"over-segmented into {k} centroids", flush=True)

    # Agglomerate the centroids, not the clips: k x k is tractable, 161k x 161k is not.
    merged = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1.0 - threshold, metric="cosine",
        linkage="average").fit_predict(C)
    labels = merged[km.labels_]
    n_spk = labels.max() + 1

    sizes = np.bincount(labels)
    print(f"{n_spk} pseudo-speakers at cosine >= {threshold}")
    print(f"  clips per speaker: median {int(np.median(sizes))}, "
          f"max {sizes.max()} ({sizes.max()/len(labels):.1%} of corpus), "
          f"singletons {(sizes == 1).sum()}")
    big = np.argsort(sizes)[::-1][:5]
    print(f"  largest 5: {[(int(b), int(sizes[b])) for b in big]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "id": pa.array(ids.tolist(), pa.string()),
        "speaker": pa.array([f"{prefix}_{v:04d}" for v in labels], pa.string()),
        "speaker_idx": pa.array(labels.tolist(), pa.int32()),
    }
    if M is not None:
        for j, name in enumerate(("snr_est", "clip_frac", "silence_frac")):
            cols[name] = pa.array(M[:, j].tolist(), pa.float32())
        print(f"  snr_est: p5 {np.percentile(M[:,0],5):.1f} dB, median "
              f"{np.median(M[:,0]):.1f}, p95 {np.percentile(M[:,0],95):.1f}")
    pq.write_table(pa.table(cols), out, compression="zstd")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "cluster", "both"], default="both")
    ap.add_argument("--data", default="data/raw/data")
    ap.add_argument("--embdir", default="out/spk_emb")
    ap.add_argument("--out", default="out/speakers.parquet")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="cosine similarity to treat as the same speaker")
    ap.add_argument("--centroids", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--prefix", default="spk", help="pseudo-speaker id prefix, e.g. 'eng'")
    args = ap.parse_args()

    if args.stage in ("embed", "both"):
        embed_shards(Path(args.data), Path(args.embdir), args.batch, args.threads,
                     args.device, args.audio_col)
    if args.stage in ("cluster", "both"):
        cluster(Path(args.embdir), Path(args.out), args.threshold, args.centroids,
                args.seed, args.prefix)


if __name__ == "__main__":
    main()
