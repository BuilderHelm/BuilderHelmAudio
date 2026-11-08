# Reproduce the local Parakeet Unified export

Targets all Apple Silicon Macs (M1 and later) with macOS 15+. Conversion requires Python 3.11+, sufficient free space for the original checkpoint (2.5 GB), a temporary extracted checkpoint and the exported models, and the pinned dependencies in `requirements.txt`. Keep this conversion environment separate from a NeMo reference environment.

Download `parakeet-unified-en-0.6b.nemo` from NVIDIA revision `fe53cd885760c96b6a5f51a0bfd362cb4584a98b`. The exporter requires archive SHA-256 `ec23ed9150c8fde49072c3e2d61678ab903dbcef389d658db833420cbc1da35b`; it does not resolve moving defaults or competitor assets.

```sh
python -m pip install -r Scripts/ParakeetUnified/requirements.txt
python Scripts/ParakeetUnified/build_release.py \
  --nemo /path/to/parakeet-unified-en-0.6b.nemo \
  --example-wav /path/to/real-mono-16khz-pcm16.wav \
  --out /path/to/new-build
```

`build_release.py` runs every step below into `<out>/work` and writes the release folder to `<out>/release`. Each step refuses existing outputs and checks its inputs' checksums. The example must be real speech lasting 0.5–15 seconds.

## What the model is

Parakeet Unified EN 0.6B is one English FastConformer (24 layers, width 1024) with an RNN-T decoder (2 × 640 LSTM predictor, joint over 1024 SentencePiece tokens plus blank 1024). The checkpoint is trained for both offline and chunked streaming inference with shared weights. This export is the offline mode: full-context attention and non-causal convolutions. The checkpoint's chunked attention (`chunked_limited_with_rc`) and Dynamic Chunked Convolution (`dcc`) settings only apply to streaming, and the exporter checks that the offline contract (`att_context_size [-1, -1]`, no convolution context override) holds.

Compared with our v3 export, the encoder has biases on every Linear and Conv1d except `linear_pos`, scales the subsampled input by √1024 (`xscaling`), and the decoder is a plain transducer: no duration head, greedy decoding with at most 10 symbols per frame.

## Steps

1. `export.py` reconstructs the source architecture from the original weights, loads component state dictionaries strictly and converts each part to an ML Program. `--precision fp16` gives FP16 encoder/predictor/joint weights and compute; the frontend and all floating-point interfaces remain FP32. `--window-seconds 15` with all parts writes a loadable `config.json`; `--window-seconds 30 --parts frontend encoder` gives the long window.
2. `merge_windows.py --split-encoder` combines the 15- and 30-second exports into `w15`/`w30` functions with Core ML constant deduplication, and bisects the FP16 encoder into two approximately equal-weight sequential programs, `Encoder` and `EncoderTail`, each below 1 GB.
3. `stage_preprocessor.py` adds `features`, `normalize_w15` and `normalize_w30` to the preprocessor and sets `staged_preprocessor: true`. The runtime computes FP32 pre-emphasis, extracts at most 401 raw log-mel frames per call, then normalizes the complete window. Original functions and weight bytes are preserved.
4. `quantize_int8.py` writes `variants/int8/`: each window's FP16 encoder compressed with weight-only, per-output-channel, symmetric linear INT8 (compute stays FP16), merged into one `w15`/`w30` encoder that is not partitioned. Its `config.json` lists the root checksums of the shared frontend, predictor, joint and vocabulary, so a downloaded variant folder is self-contained.

## Release layout

```
release/
  config.json  vocab.json  merge-record.json
  Preprocessor.mlpackage   (FP32; w15, w30, features, normalize_w15, normalize_w30)
  Encoder.mlpackage        (FP16; w15, w30; first half)
  EncoderTail.mlpackage    (FP16; w15, w30; second half)
  Predictor.mlpackage      (FP16)
  Joint.mlpackage          (FP16)
  variants/int8/
    config.json  variant-record.json
    Encoder.mlpackage      (INT8 weights, FP16 compute; w15, w30)
```

About 1.2 GB for FP16 and about 0.6 GB for the INT8 encoder plus the shared 21 MB.

## Validation

The export record's example tensors are a Torch reference only, not independent NeMo parity evidence. A new exporter revision must be validated against captures from the original checkpoint: numerical error envelopes, valid lengths and padding, full greedy decode sequences and WER, on held-out audio and on 15–30-second recordings, before deployment. Weight-only per-channel INT8 is the qualified scheme; per-block INT8 moves most operations off the Neural Engine, and post-training INT8 activations were rejected on accuracy.

M5 is the available Mac validation machine. Models are distributed as portable `.mlpackage` files and compiled locally; no chip-specific compiled model or requirement is introduced. Other-chip performance is unmeasured hardware coverage.

## License

The original and derived weights are governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). Exporter and runtime implementation are covered by this repository's license.
