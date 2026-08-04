"""Convert KoelLabs/xlsr-english-01 to an ONNX model sherpa-onnx can load.

Why this model: evaluating English TTS with the Ghana phoneme ASR was a mistake. English is
not among its 42 training languages, and real human recordings score 70% unit error against a
canonical reference — so any TTS number sitting on top of that floor is unreadable. KoelLabs
is a wav2vec2 CTC phoneme recogniser trained specifically on *accented* English (L2-ARCTIC,
EpaDB, Speech Ocean), reporting ~19% PER, and it emits IPA. That makes it a like-for-like
instrument against IPA targets rather than a word-level proxy.

Two things have to be right for sherpa-onnx to load it through `from_omnilingual_asr_ctc`,
which is a generic wav2vec2-CTC loader:

  graph signature   raw waveform (N, samples) float32 -> logits (N, frames, vocab). No
                    attention mask input, since sherpa does not supply one.
  normalisation     the HF processor sets do_normalize=True, so the encoder expects zero-mean
                    unit-variance audio. sherpa feeds raw samples, so the normalisation is
                    baked into the graph instead of being left as an unstated precondition.
                    This is the same class of bug that cost this project a day: a front-end
                    difference that produces plausible output rather than an error.

Note the licence: KoelLabs is AGPL-3.0. It is used here as an offline measurement tool and is
deliberately not a dependency of the published inference package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


class Normalized(torch.nn.Module):
    """Zero-mean unit-variance per utterance, then the CTC model.

    Mirrors transformers' `zero_mean_unit_var_norm`, which uses var + 1e-7 rather than a plain
    epsilon-on-std; matching it exactly matters because the encoder was trained on its output.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-7)
        return self.model(x).logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="KoelLabs/xlsr-english-01")
    ap.add_argument("--out", default="out/koel-en")
    ap.add_argument("--opset", type=int, default=14)
    ap.add_argument("--quantize", action="store_true",
                    help="also write an int8 model, as sherpa deployments usually want")
    args = ap.parse_args()

    from transformers import AutoModelForCTC, AutoProcessor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model).eval()

    fe = proc.feature_extractor
    if not getattr(fe, "do_normalize", False):
        print("note: processor does not normalise; the wrapper still will, which would be wrong")
    sr = getattr(fe, "sampling_rate", 16000)
    print(f"sample rate {sr}, vocab {model.config.vocab_size}")

    vocab = proc.tokenizer.get_vocab()
    by_id = {i: s for s, i in vocab.items()}
    # sherpa reads tokens.txt as "<symbol> <id>". A symbol containing a space cannot survive
    # that format; none do here, but assert rather than emit a silently broken table.
    with open(out / "tokens.txt", "w", encoding="utf-8") as fh:
        for i in range(len(by_id)):
            sym = by_id[i]
            assert " " not in sym, f"symbol {sym!r} contains a space"
            fh.write(f"{sym} {i}\n")
    print(f"tokens.txt: {len(by_id)} symbols")

    wrapper = Normalized(model).eval()
    dummy = torch.randn(1, sr * 3)
    onnx_path = out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path), opset_version=args.opset,
            input_names=["x"], output_names=["logits"],
            dynamic_axes={"x": {0: "N", 1: "num_samples"},
                          "logits": {0: "N", 1: "num_frames"}},
        )

    # sherpa-onnx reads these to decide how to feed the model.
    import onnx
    m = onnx.load(str(onnx_path))
    while len(m.metadata_props):
        m.metadata_props.pop()
    for k, v in {
        "vocab_size": model.config.vocab_size, "model_type": "omnilingual-asr",
        "version": "1", "sample_rate": sr, "model_author": "KoelLabs",
        "comment": "xlsr-english-01 English IPA phoneme CTC, waveform normalisation baked in",
    }.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, str(onnx_path))
    print(f"model.onnx: {onnx_path.stat().st_size/1e6:.0f} MB")

    # Parity check: the ONNX graph must agree with the torch model through the processor.
    import numpy as np
    import onnxruntime as ort
    wav = torch.randn(1, sr * 2)
    with torch.no_grad():
        want = wrapper(wav).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": wav.numpy()})[0]
    diff = float(np.abs(want - got).max())
    agree = (want.argmax(-1) == got.argmax(-1)).mean()
    print(f"onnx vs torch: max|Δlogit| {diff:.2e}, argmax agreement {agree:.4%}")
    if agree < 0.999:
        raise SystemExit("export changed predictions; do not use this model")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out / "model.int8.onnx"
        quantize_dynamic(model_input=str(onnx_path), model_output=str(q),
                         op_types_to_quantize=["MatMul"], weight_type=QuantType.QUInt8)
        print(f"model.int8.onnx: {q.stat().st_size/1e6:.0f} MB")

    (out / "meta.json").write_text(json.dumps(
        {"source": args.model, "licence": "AGPL-3.0", "sample_rate": sr,
         "vocab_size": model.config.vocab_size,
         "sherpa_loader": "OfflineRecognizer.from_omnilingual_asr_ctc",
         "note": "waveform normalisation is inside the graph; feed raw samples"}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
