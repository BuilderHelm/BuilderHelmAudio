"""Represent the 110M DFT convolutions as height-one 2D convolutions.

Requires coremltools==9.0. Usage: python lift-preprocessor-dft.py source.mlmodel output.mlmodel
Weights retain their exact bytes; only singleton dimensions and convolution metadata change.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import coremltools as ct

MIL = ct.proto.MIL_pb2


def tensor_type(dtype, shape):
    value = MIL.ValueType()
    value.tensorType.dataType = dtype
    value.tensorType.rank = len(shape)
    for size in shape:
        value.tensorType.dimensions.add().constant.size = size
    return value


def operation(kind, name, shape, inputs=None, dtype=MIL.FLOAT32):
    op = MIL.Operation()
    op.type = kind
    output = op.outputs.add()
    output.name = name
    output.type.CopyFrom(tensor_type(dtype, shape))
    op.attributes["name"].type.CopyFrom(tensor_type(MIL.STRING, []))
    op.attributes["name"].immediateValue.tensor.strings.values.append(name)
    for key, names in (inputs or {}).items():
        for name in names:
            op.inputs[key].arguments.add().name = name
    return op


def integers(name, values):
    op = operation("const", name, [len(values)], dtype=MIL.INT32)
    op.attributes["val"].type.CopyFrom(op.outputs[0].type)
    op.attributes["val"].immediateValue.tensor.ints.values.extend(values)
    return op


def lift(spec):
    changed = 0
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            nodes = {v.name: op for op in block.operations for v in op.outputs}
            for name in ["mel_dft_real", "mel_dft_imag"]:
                op = nodes[name]
                shape = [d.constant.size for d in op.outputs[0].type.tensorType.dimensions]
                if shape != [257, 1, 512]:
                    raise ValueError(f"Unexpected DFT weight shape: {shape}")
                shape.insert(2, 1)
                op.outputs[0].type.CopyFrom(tensor_type(MIL.FLOAT32, shape))
                op.attributes["val"].type.CopyFrom(op.outputs[0].type)
            for name in ["real", "imag"]:
                for key, values in [("strides", [1, 160]), ("pad", [0, 0, 0, 0]), ("dilations", [1, 1])]:
                    constant = nodes[nodes[name].inputs[key].arguments[0].name]
                    constant.CopyFrom(integers(constant.outputs[0].name, values))
            result = []
            for original in block.operations:
                name = original.outputs[0].name
                if original.type != "conv" or name not in ["real", "imag"]:
                    result.append(deepcopy(original))
                    continue
                if name == "real":
                    shape = [d.constant.size for d in nodes["x"].outputs[0].type.tensorType.dimensions]
                    shape.insert(2, 1)
                    result.extend([
                        integers("dft_height_axis", [2]),
                        operation("expand_dims", "dft_waveform_2d", shape,
                                  {"x": ["x"], "axes": ["dft_height_axis"]}),
                    ])
                output_type = deepcopy(original.outputs[0].type)
                shape = [d.constant.size for d in output_type.tensorType.dimensions]
                shape.insert(2, 1)
                conv = deepcopy(original)
                conv.outputs[0].name = name + "_2d"
                conv.outputs[0].type.CopyFrom(tensor_type(MIL.FLOAT32, shape))
                conv.attributes["name"].immediateValue.tensor.strings.values[0] = name + "_2d"
                conv.inputs["x"].arguments[0].name = "dft_waveform_2d"
                squeezed = operation("squeeze", name, [], {"x": [name + "_2d"], "axes": ["dft_height_axis"]})
                squeezed.outputs[0].type.CopyFrom(output_type)
                result.extend([conv, squeezed])
                changed += 1
            del block.operations[:]
            block.operations.extend(result)
    if changed != 8:
        raise ValueError(f"Expected eight DFT convolutions, found {changed}")
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
    spec = lift(ct.utils.load_spec(str(args.source)))
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
