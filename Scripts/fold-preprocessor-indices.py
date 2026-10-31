"""Fold the 110M preprocessor's static sample-index range before CoreML compilation.

Requires coremltools==9.0. Usage: python fold-preprocessor-indices.py source.mlmodel output.mlmodel
Weights, normalization, and public model inputs/outputs are unchanged.
"""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import coremltools as ct


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
    spec = ct.utils.load_spec(str(args.source))
    folded = []
    for function_name, function in spec.mlProgram.functions.items():
        for block in function.block_specializations.values():
            constants = {op.outputs[0].name: op.attributes["val"] for op in block.operations if op.type == "const"}
            for op in block.operations:
                if op.type != "range_1d":
                    continue
                bounds = {key: constants[value.arguments[0].name].immediateValue.tensor.ints.values[0]
                          for key, value in op.inputs.items()}
                if function_name != "w20" or bounds != {"start": 0, "end": 320_000, "step": 1}:
                    raise ValueError(f"Unexpected range in {function_name}: {bounds}")
                output = deepcopy(op.outputs[0])
                name = deepcopy(op.attributes["name"])
                op.Clear()
                op.type = "const"
                op.outputs.add().CopyFrom(output)
                op.attributes["name"].CopyFrom(name)
                value = op.attributes["val"]
                value.type.CopyFrom(output.type)
                value.immediateValue.tensor.ints.values.extend(range(320_000))
                folded.append(function_name)
    if folded != ["w20"]:
        raise ValueError(f"Expected one w20 sample-index range, found {folded}")
    data = spec.SerializeToString(deterministic=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"{len(data)} bytes; SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
