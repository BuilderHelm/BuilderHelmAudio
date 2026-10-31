"""Bound the 110M preprocessor's DFT convolutions without changing window normalization.

Requires coremltools==9.0. Usage: python tile-preprocessor-dft.py source.mlmodel output.mlmodel
Uses the original weight.bin unchanged. Every DFT frame retains its exact 512-sample input.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import coremltools as ct

MIL = ct.proto.MIL_pb2


def tensor_type(dtype, shape):
    result = MIL.ValueType()
    result.tensorType.dataType = dtype
    result.tensorType.rank = len(shape)
    for size in shape:
        result.tensorType.dimensions.add().constant.size = size
    return result


def operation(kind, name, shape, inputs=None, dtype=MIL.FLOAT32):
    op = MIL.Operation()
    op.type = kind
    output = op.outputs.add()
    output.name = name
    output.type.CopyFrom(tensor_type(dtype, shape))
    op.attributes["name"].type.CopyFrom(tensor_type(MIL.STRING, []))
    op.attributes["name"].immediateValue.tensor.strings.values.append(name)
    for key, names in (inputs or {}).items():
        for value in names:
            op.inputs[key].arguments.add().name = value
    return op


def constant(name, values, shape, dtype=MIL.INT32):
    op = operation("const", name, shape, dtype=dtype)
    value = op.attributes["val"]
    value.type.CopyFrom(op.outputs[0].type)
    if dtype == MIL.BOOL:
        value.immediateValue.tensor.bools.values.extend(values)
    else:
        value.immediateValue.tensor.ints.values.extend(values)
    return op


def tile(spec):
    replacements = 0
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            result = []
            for original in block.operations:
                name = original.outputs[0].name
                if original.type != "conv" or name not in ("real", "imag"):
                    result.append(deepcopy(original))
                    continue
                dimensions = original.outputs[0].type.tensorType.dimensions
                frames = dimensions[-1].constant.size
                if frames <= 401:  # Preserve the already bounded four-second function.
                    result.append(deepcopy(original))
                    continue
                channels = dimensions[1].constant.size
                pieces = []
                for start in range(0, frames, 400):
                    count = min(400, frames - start)
                    prefix = f"{name}_tile_{start}"
                    first_sample = start * 160
                    samples = (count - 1) * 160 + 512
                    result.extend([
                        constant(prefix + "_begin", [0, 0, first_sample], [3]),
                        constant(prefix + "_end", [1, 1, first_sample + samples], [3]),
                        operation("slice_by_index", prefix + "_input", [1, 1, samples], {
                            "x": [original.inputs["x"].arguments[0].name],
                            "begin": [prefix + "_begin"], "end": [prefix + "_end"],
                        }),
                    ])
                    conv = deepcopy(original)
                    conv.outputs[0].name = prefix
                    conv.outputs[0].type.CopyFrom(tensor_type(MIL.FLOAT32, [1, channels, count]))
                    conv.attributes["name"].immediateValue.tensor.strings.values[0] = prefix
                    conv.inputs["x"].arguments[0].name = prefix + "_input"
                    result.append(conv)
                    pieces.append(prefix)
                result.extend([
                    constant(name + "_tile_axis", [2], []),
                    constant(name + "_tile_interleave", [False], [], MIL.BOOL),
                    operation("concat", name, [1, channels, frames], {
                        "values": pieces, "axis": [name + "_tile_axis"],
                        "interleave": [name + "_tile_interleave"],
                    }),
                ])
                replacements += 1
            del block.operations[:]
            block.operations.extend(result)
    if replacements != 6:
        raise ValueError(f"Expected six wide DFT convolutions, found {replacements}")
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask-probe-output", type=Path)
    args = parser.parse_args()
    expected = "9def4672829e834402d523bcd4ea69366fde9873f669418a80466b5480a3f2ef"
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != expected:
        raise ValueError("Expected the pinned 110M v1.1.0 preprocessor specification")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mask_probe_output is not None and args.mask_probe_output.exists():
        raise FileExistsError(args.mask_probe_output)
    spec = tile(ct.utils.load_spec(str(args.source)))
    data = spec.SerializeToString(deterministic=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    if args.mask_probe_output is not None:
        probe = deepcopy(spec)
        for name in list(probe.mlProgram.functions):
            if name != "w20":
                del probe.mlProgram.functions[name]
        descriptor = deepcopy(next(f for f in probe.description.functions if f.name == "w20"))
        del probe.description.functions[:]
        probe.description.functions.append(descriptor)
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
    print(f"{len(data)} bytes; SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
