"""Add bounded DFT and full-window normalization functions without changing weights.

Requires Python 3.11 and coremltools==9.0. Source and destination are complete V3
model directories. The source remains immutable; the destination must not exist.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import coremltools as ct


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shape(value, dimensions):
    tensor = value.type.tensorType
    tensor.rank = len(dimensions)
    del tensor.dimensions[:]
    for size in dimensions:
        tensor.dimensions.add().constant.size = size


def prune(block, outputs, inputs):
    live, selected = set(outputs), []
    for op in reversed(block.operations):
        if any(value.name in live - set(inputs) for value in op.outputs):
            selected.append(deepcopy(op))
            for value in op.inputs.values():
                live.update(argument.name for argument in value.arguments)
    defined = set(inputs) | {value.name for op in selected for value in op.outputs}
    if live - defined:
        raise ValueError(f"Unbound stage values: {sorted(live - defined)}")
    del block.operations[:]
    block.operations.extend(reversed(selected))
    del block.outputs[:]
    block.outputs.extend(outputs)


def stage(spec):
    descriptions = {f.name: deepcopy(f) for f in spec.description.functions}
    if set(descriptions) != {"w15", "w30"}:
        raise ValueError("Expected the original V3 w15/w30 frontend")
    features = deepcopy(spec.mlProgram.functions["w15"])
    waveform = deepcopy(features.inputs[0])
    waveform.name = "waveform"
    shape(waveform, [1, 1, 64512])
    del features.inputs[:]
    features.inputs.append(waveform)
    for block in features.block_specializations.values():
        for op in block.operations:
            for value in op.inputs.values():
                for argument in value.arguments:
                    if argument.name == "padded":
                        argument.name = "waveform"
            for value in op.outputs:
                if value.name == "logged":
                    value.name = "raw_mel"
        prune(block, ["raw_mel"], ["waveform"])
        for op in block.operations:
            for value in op.outputs:
                dimensions = value.type.tensorType.dimensions
                if dimensions and dimensions[-1].constant.size == 1501:
                    dimensions[-1].constant.size = 401
    spec.mlProgram.functions["features"].CopyFrom(features)
    description = spec.description.functions.add()
    description.name = "features"
    for collection, name, dimensions in [
        (description.input, "waveform", [1, 1, 64512]),
        (description.output, "raw_mel", [1, 128, 401]),
    ]:
        value = collection.add()
        value.name = name
        value.type.multiArrayType.shape.extend(dimensions)
        value.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    for name, original in descriptions.items():
        normalizer = deepcopy(spec.mlProgram.functions[name])
        frames = original.output[0].type.multiArrayType.shape[-1]
        normalizer.inputs[0].name = "raw_mel"
        shape(normalizer.inputs[0], [1, 128, frames])
        for block in normalizer.block_specializations.values():
            for op in block.operations:
                for value in op.inputs.values():
                    for argument in value.arguments:
                        if argument.name == "logged":
                            argument.name = "raw_mel"
            prune(block, ["mel", "mel_length"], ["raw_mel", "audio_length"])
        spec.mlProgram.functions["normalize_" + name].CopyFrom(normalizer)
        description = deepcopy(original)
        description.name = "normalize_" + name
        description.input[0].name = "raw_mel"
        del description.input[0].type.multiArrayType.shape[:]
        description.input[0].type.multiArrayType.shape.extend([1, 128, frames])
        spec.description.functions.append(description)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    if args.destination.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("Destination must be outside the source directory")
    config = json.loads((args.source / "config.json").read_text())
    if config["architecture"] != "parakeet-tdt-0.6b-v3" or config.get("staged_preprocessor"):
        raise ValueError("Expected an unstaged V3 export")
    for relative, expected in config["files"].items():
        if sha(args.source / relative) != expected:
            raise ValueError("Source checksum mismatch: " + relative)
    relative = "Preprocessor.mlpackage/Data/com.apple.CoreML/model.mlmodel"
    spec = stage(ct.utils.load_spec(str(args.source / relative)))
    shutil.copytree(args.source, args.destination)
    (args.destination / relative).write_bytes(spec.SerializeToString(deterministic=True))
    config["staged_preprocessor"] = True
    config["files"][relative] = sha(args.destination / relative)
    (args.destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path, expected in config["files"].items():
        if sha(args.destination / path) != expected:
            raise ValueError("Destination checksum mismatch: " + path)
    print(args.destination)


if __name__ == "__main__":
    main()
