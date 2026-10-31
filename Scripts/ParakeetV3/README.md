# Reproduce the local v3 export

Targets all Apple Silicon Macs (M1 and later) with macOS 15+. Conversion requires Python 3.11+, sufficient free space for the original checkpoint, a temporary extracted checkpoint and the exported models, and the pinned dependencies in `requirements.txt`. Keep this conversion environment separate from a NeMo reference environment.

Download `parakeet-tdt-0.6b-v3.nemo` from NVIDIA revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. The exporter requires archive SHA-256 `3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d`; it does not resolve moving defaults or competitor assets.

```sh
python -m pip install -r Scripts/ParakeetV3/requirements.txt
python Scripts/ParakeetV3/export.py \
  --nemo /path/to/parakeet-tdt-0.6b-v3.nemo \
  --example-wav /path/to/real-mono-16khz-pcm16.wav \
  --precision fp32 --out /path/to/new-fp32-export
```

Use `--precision fp16` for the unquantized FP16 encoder/predictor/joint baseline. The frontend and all floating-point interfaces remain FP32. The example must be real speech lasting 0.5–15 seconds. Existing output directories are rejected. `--parts` can export individual components; only an all-component export receives a loadable SDK config.

The exporter reconstructs the source architecture from original weights, loads component state dictionaries strictly, retains full token/duration logits and writes package checksums, source attribution, an export record and a Torch example tensor reference. The example reference is not independent NeMo parity evidence. A new exporter revision must be validated against independent original-checkpoint captures, including numerical error envelopes, valid lengths/padding, teacher-forced state trajectories, full decode sequences and WER before deployment.

Load the completed local folder using `.experimentalParakeetV3`; see `Documentation/ExperimentalParakeetV3.md`. A single-function export has a fixed 15-second window; the shared-weight release export adds a 30-second function. The SDK handles longer offline recordings with bounded quiet-pause segmentation, overlapping token consensus and bounded blank-window recovery. It advertises neither CTC nor timing outputs and does not publish model files.

NVIDIA's original/derived weights retain CC BY 4.0 attribution. Exporter/runtime implementation is covered by this repository's license. The architecture code reuses BuilderHelm's own previously validated PyTorch encoder math with the v3 dimensions and bias/LSTM/vocabulary contracts.

M5 is the available Mac validation machine. Models are distributed as portable `.mlpackage` files and compiled locally; no M5-only compiled model or chip requirement is introduced. Other-chip performance is unmeasured hardware coverage, not a requirement to obtain M1 hardware.

For the shared-weight 15/30-second export, export a full FP16 15-second folder and a 30-second frontend/encoder folder, then merge them:

```sh
python Scripts/ParakeetV3/export.py --nemo /path/model.nemo --example-wav /path/real-speech.wav --precision fp16 --window-seconds 30 --parts frontend encoder --out /path/long-export
python Scripts/ParakeetV3/merge_windows.py --short /path/short-export --long /path/long-export --out /path/combined-export
```

The merger verifies source manifests and uses Core ML constant deduplication. The short function remains the default; the runtime loads the 30-second encoder only for longer requests. Final model qualification and a public pinned descriptor are required before advertising a new export as released.
