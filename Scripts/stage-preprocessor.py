"""Add bounded feature extraction and full-window normalization functions to the 110M export.

Requires coremltools==9.0. Usage: python stage-preprocessor.py source.mlmodel output.mlmodel
Set staged_preprocessor=true in config.json. The existing functions and weight.bin are preserved.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import coremltools as ct


def shape(named_value, dimensions):
    tensor = named_value.type.tensorType
    tensor.rank = len(dimensions)
    del tensor.dimensions[:]
    for size in dimensions:
        tensor.dimensions.add().constant.size = size


def prune(block, outputs, inputs):
    live = set(outputs)
    selected = []
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
    if set(descriptions) != {"w4", "w8", "w15", "w20"}:
        raise ValueError("Expected the four published window functions")
    features = deepcopy(spec.mlProgram.functions["w4"])
    waveform = deepcopy(features.inputs[0])
    waveform.name = "waveform"
    shape(waveform, [1, 1, 64_512])
    del features.inputs[:]
    features.inputs.append(waveform)
    for block in features.block_specializations.values():
        for op in block.operations:
            for value in op.inputs.values():
                for argument in value.arguments:
                    if argument.name == "x":
                        argument.name = "waveform"
            if op.outputs[0].name == "mel_1":
                op.outputs[0].name = "raw_mel"
        prune(block, ["raw_mel"], ["waveform"])
    spec.mlProgram.functions["features"].CopyFrom(features)
    description = spec.description.functions.add()
    description.name = "features"
    feature_input = description.input.add()
    feature_input.name = "waveform"
    feature_input.type.multiArrayType.shape.extend([1, 1, 64_512])
    feature_input.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    feature_output = description.output.add()
    feature_output.name = "raw_mel"
    feature_output.type.multiArrayType.shape.extend([1, 80, 401])
    feature_output.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    for name, original_description in descriptions.items():
        normalizer = deepcopy(spec.mlProgram.functions[name])
        frames = original_description.output[0].type.multiArrayType.shape[-1]
        normalizer.inputs[0].name = "raw_mel"
        shape(normalizer.inputs[0], [1, 80, frames])
        for block in normalizer.block_specializations.values():
            for op in block.operations:
                for value in op.inputs.values():
                    for argument in value.arguments:
                        if argument.name == "mel_1":
                            argument.name = "raw_mel"
            prune(block, ["mel", "mel_length"], ["raw_mel", "audio_length"])
        spec.mlProgram.functions["normalize_" + name].CopyFrom(normalizer)
        description = deepcopy(original_description)
        description.name = "normalize_" + name
        description.input[0].name = "raw_mel"
        del description.input[0].type.multiArrayType.shape[:]
        description.input[0].type.multiArrayType.shape.extend([1, 80, frames])
        spec.description.functions.append(description)
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    expected = "9def4672829e834402d523bcd4ea69366fde9873f669418a80466b5480a3f2ef"
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the pinned 110M v1.1.0 preprocessor specification")
    if args.output.exists():
        raise FileExistsError(args.output)
    data = stage(ct.utils.load_spec(str(args.source))).SerializeToString(deterministic=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"{len(data)} bytes; SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
