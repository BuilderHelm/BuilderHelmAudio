"""Fold the 110M preprocessor's static sample-index range before CoreML compilation.

Requires coremltools==9.0. Usage:
  python fold-preprocessor-indices.py source.mlmodel output.mlmodel sample_indices.bin
The new blob belongs at weights/sample_indices.bin beside the output specification.
Trained weights, normalization, and public model inputs/outputs are unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import coremltools as ct
from coremltools.libmilstoragepython import _BlobStorageReader, _BlobStorageWriter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("indices", type=Path)
    parser.add_argument("--mask-probe-output", type=Path)
    args = parser.parse_args()
    expected = "9def4672829e834402d523bcd4ea69366fde9873f669418a80466b5480a3f2ef"
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the pinned 110M v1.1.0 preprocessor specification")
    for path in [args.output, args.indices, args.mask_probe_output]:
        if path is not None and path.exists():
            raise FileExistsError(path)
    spec = ct.utils.load_spec(str(args.source))
    folded = []
    for function_name, function in spec.mlProgram.functions.items():
        for block in function.block_specializations.values():
            constants = {op.outputs[0].name: op.attributes["val"] for op in block.operations if op.type == "const"}
            for op in list(block.operations):
                if op.type != "range_1d":
                    continue
                bounds = {key: constants[value.arguments[0].name].immediateValue.tensor.ints.values[0]
                          for key, value in op.inputs.items()}
                if function_name != "w20" or bounds != {"start": 0, "end": 320_000, "step": 1}:
                    raise ValueError(f"Unexpected range in {function_name}: {bounds}")
                name = op.outputs[0].name
                users = [other for other in block.operations
                         if any(arg.name == name for value in other.inputs.values() for arg in value.arguments)]
                if len(users) != 1 or users[0].type != "expand_dims" or users[0].outputs[0].name != "var_23":
                    raise ValueError("Expected the sample-index range's single expand_dims consumer")
                target = users[0]
                output, name_attribute = deepcopy(target.outputs[0]), deepcopy(target.attributes["name"])
                args.indices.parent.mkdir(parents=True, exist_ok=True)
                indices = np.arange(320_000, dtype=np.int32)
                writer = _BlobStorageWriter(str(args.indices))
                offset = writer.write_int32_data(indices)
                del writer
                # Validate every value, including the regions past 65,536 samples.
                np.testing.assert_array_equal(_BlobStorageReader(str(args.indices)).read_int32_data(offset), indices)
                target.Clear()
                target.type = "const"
                target.outputs.add().CopyFrom(output)
                target.attributes["name"].CopyFrom(name_attribute)
                value = target.attributes["val"]
                value.type.CopyFrom(output.type)
                value.blobFileValue.fileName = "@model_path/weights/sample_indices.bin"
                value.blobFileValue.offset = offset
                for i, candidate in enumerate(block.operations):
                    if candidate.outputs[0].name == name:
                        del block.operations[i]
                        break
                folded.append(function_name)
    if folded != ["w20"]:
        raise ValueError(f"Expected one w20 sample-index range, found {folded}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(spec.SerializeToString(deterministic=True))
    if args.mask_probe_output:
        # A test-only view of this same input mask, stopping before preprocessing/normalization.
        probe = deepcopy(spec)
        for name in list(probe.mlProgram.functions):
            if name != "w20":
                del probe.mlProgram.functions[name]
        descriptors = [deepcopy(f) for f in probe.description.functions if f.name == "w20"]
        del probe.description.functions[:]
        probe.description.functions.extend(descriptors)
        descriptor = probe.description.functions[0]
        del descriptor.output[:]
        output = descriptor.output.add()
        output.name = "audio_1"
        output.type.multiArrayType.shape.extend([1, 320_000])
        output.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
        block = next(iter(probe.mlProgram.functions["w20"].block_specializations.values()))
        end = next(i for i, op in enumerate(block.operations) if op.outputs[0].name == "audio_1")
        del block.operations[end + 1:]
        del block.outputs[:]
        block.outputs.append("audio_1")
        args.mask_probe_output.parent.mkdir(parents=True, exist_ok=True)
        args.mask_probe_output.write_bytes(probe.SerializeToString(deterministic=True))
    for path in [args.output, args.indices]:
        data = path.read_bytes()
        print(f"{path.name}: {len(data)} bytes; SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
