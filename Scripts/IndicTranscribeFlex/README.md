# Indic-Transcribe Flex Core ML conversion

The exporter uses the official Bodhan checkpoint pinned in `download.py`. Accept
the model terms in Hugging Face before downloading. Run these commands from the
BuilderHelmAudio repository root on a Mac with Python 3.12 and uv.

```sh
snapshot="$(uv run --no-project --python 3.12 --with huggingface-hub \
  python Scripts/IndicTranscribeFlex/download.py)"
requirements=Scripts/IndicTranscribeFlex/requirements.txt
output=/path/to/flex-coreml
for component in frontend encoder decoder; do
  uv run --no-project --python 3.12 --with-requirements "$requirements" \
    python Scripts/IndicTranscribeFlex/export.py "$snapshot" "$output" --component "$component"
done
uv run --no-project --python 3.12 --with-requirements "$requirements" \
  python Scripts/IndicTranscribeFlex/tokenizer_assets.py "$snapshot" "$output/tokenizer.json"
```

The frontend uses FP32 and the encoder and decoder use FP16 Core ML compute.
The exported tokenizer includes all 27 explicit languages and their three output modes.
The default decoder recomputes the prefix for parity checks. Export
`--component decoder_cached` to try the experimental Core ML stateful decoder.
The checker and SDK command both accept `--cached` to select it.

Use the same short recording for both output modes. The checker compares the
frontend numerically and compares the final greedy transcript with Bodhan's
official runner. Use `--language auto --mode native` to compare the detected
language and native transcript. Explicit language checks accept the codes in the
exported tokenizer.

The Core ML Python prediction binding can abort while loading these packages
with Python 3.12. Use Python 3.11 and PyTorch 2.7 for the comparison below.

```sh
for mode in native romanized; do
  uv run --no-project --python 3.11 --with coremltools==9.0 \
    --with torch==2.7.0 --with torchaudio==2.7.0 --with transformers==4.57.6 \
    --with huggingface-hub --with sentencepiece --with soundfile \
    python Scripts/IndicTranscribeFlex/check.py "$snapshot" "$output" hindi.wav \
      --language hi --mode "$mode"
done
```

To run the exported model through BuilderHelmAudio on the Mac:

```sh
swift run bhaudio indic-transcribe hindi.wav --model-dir "$output" --language hi --mode romanized
```

This local prototype accepts clips up to 30 seconds and reports the language
and transcript as JSON. Try `--mode native` for Devanagari and
`--language auto` to exercise language detection.

Built with Indic-Transcribe-Flex from Bodhan AI / AI4Bharat.

Bodhan's Indic Open Model License governs the converted model. Preserve the
license file with the packages and credit them as “Built with Indic-Transcribe
Flex from Bodhan AI / AI4Bharat.” The SDK inference code is separate from the
model weights. For the published stateful export, use `--cached` or
`IndicTranscribeFlex(modelDirectory:useCachedDecoder:)` with
`useCachedDecoder: true`.
