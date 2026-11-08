# V3 artifact variants (in development)

The published `.parakeetTdtV3` remains the unquantized FP16 model and uses its existing
immutable manifest and storage directory. There is no published quantized descriptor
in this change. A quantized candidate needs independent accuracy, end-to-end speed
and device qualification before it can enter `AsrModel.all` or become downloadable.

## One repository, independent downloads

The proposed model repository layout keeps the current FP16 files at the root and
adds qualified alternative encoder/config files under `variants/<format>/`. Each
descriptor pins a full repository commit and lists only the files it needs. Shared
FP16 predictor, joint, frontend and vocabulary files can keep their root paths;
users download only the selected descriptor, without needing the other encoder.
Shared source files are copied into each selected model's local folder rather than
coupling the variants' removal lifecycles.

`AsrModel.File.path` is the path inside the local model folder.
`repositoryPath` is its source path within the pinned repository, defaulting to
`path` for all existing manifests. A future quantized descriptor can therefore map
`variants/palette6/config.json` to local `config.json`, while mapping a shared
`Predictor.mlpackage` to its original local path. The runtime still receives a
self-contained model folder, with its own config and package checksums.

Descriptors with an explicit `variant` use
`Models/<repository>/Variants/<revision>/<variant>/`. Existing descriptors with
nil `variant` retain `Models/<repository>/<revision>/`. These are sibling trees:
removing either FP16 or a quantized variant cannot recursively remove the other.
Verification markers remain local to the selected descriptor directory. Compiled
models are keyed by the package checksums in each variant's own `config.json` (or
its content, if the config does not list every package file) and the OS build, so a
variant's re-quantized encoder compiles separately, while a shared FP16 predictor
or joint with identical checksums reuses one compiled copy. Removing a variant does
not delete compiled models.

Variant identifiers are maintained by the SDK's internal descriptors, not arbitrary
user input. A variant is a storage/manifest identity; it does not bypass the runtime's
precision/config validation or establish model qualification. Weight storage names
must distinguish FP16, linear INT8 and palette indices; palette compression does
not imply integer activation arithmetic.

## Qualification before adding an alternative

1. Verify actual compressed tensors, source attribution, portable package size and
   immutable file hashes. Preserve the current FP16 baseline.
2. Screen real encoder execution, then run the existing pooled/per-language,
   long-audio, silence, failure and lifecycle gates on the deployed model.
3. Benchmark actual SDK file/PCM transcription for short, mixed and long audio
   against FP16 and a pinned Fluid comparator. Component speed or smaller disk
   size alone is insufficient.
4. Complete independent confirmation, physical-device checks, CI and a fresh
   consumer check before publishing a new descriptor. FP16 remains default.
