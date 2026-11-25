"""Export the official Flex checkpoint as macOS 15 Core ML programs."""

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from download import REVISION


class Frontend(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.hop = source.hop_length
        self.padding = source.n_fft // 2
        self.preemphasis = source.preemph
        window = F.pad(source.window, ((source.n_fft - source.win_length) // 2,) * 2)
        samples = torch.arange(source.n_fft, dtype=torch.float64)
        frequencies = torch.arange(source.n_fft // 2 + 1, dtype=torch.float64)
        phase = (2 * torch.pi / source.n_fft) * frequencies[:, None] * samples[None, :]
        self.register_buffer("dft", (torch.cat((phase.cos(), phase.sin())) * window).float()[:, None])
        self.register_buffer("mel", source.fb)
        self.bins = frequencies.numel()

    def forward(self, audio):
        signal = torch.cat((audio[:, :1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1)
        signal = F.pad(signal[:, None], (self.padding, self.padding), mode="reflect")
        spectrum = F.conv1d(signal, self.dft, stride=self.hop)
        power = spectrum[:, :self.bins].square() + spectrum[:, self.bins:].square()
        mel = (self.mel @ power + 2.0**-24).log()
        centered = mel - mel.mean(dim=-1, keepdim=True)
        variance = centered.square().sum(dim=-1, keepdim=True) / (mel.size(-1) - 1)
        return centered / (variance.sqrt() + 1e-5)


class Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.get_encoder()
        self.encoder.pos_enc.extend_pe(376, torch.device("cpu"), torch.float32)

    def forward(self, features):
        mask = torch.ones_like(features[:, 0, :], dtype=torch.int32)
        return self.encoder(features, attention_mask=mask).last_hidden_state


def split_heads(layer, values):
    return values.reshape(1, -1, layer.h, layer.d_k).transpose(1, 2)


def project_kv(layer, states):
    keys = split_heads(layer, layer.key_net(states)) / layer.attn_scale
    values = split_heads(layer, layer.value_net(states))
    return keys, values


def attention(layer, hidden, keys, values, mask=None):
    queries = split_heads(layer, layer.query_net(hidden)) / layer.attn_scale
    scores = queries @ keys.transpose(-1, -2)
    if mask is not None:
        scores = scores + mask
    context = (scores.softmax(dim=-1) @ values).transpose(1, 2)
    return layer.out_projection(context.reshape(1, -1, layer.h * layer.d_k))


class Decoder(nn.Module):
    """Reference decoder with explicit attention for Core ML graph conversion."""

    def __init__(self, model):
        super().__init__()
        self.decoder = model.get_decoder()
        self.projection = model.lm_head

    def forward(self, ids, encoder_states, positions, mask):
        embedding = self.decoder.embedding
        hidden = embedding.layer_norm(
            embedding.token_embedding(ids) + embedding.position_embedding(positions)
        )
        for layer in self.decoder.layers:
            normed = layer.layer_norm_1(hidden)
            self_keys, self_values = project_kv(layer.first_sub_layer, normed)
            hidden = hidden + attention(layer.first_sub_layer, normed, self_keys, self_values, mask)
            cross_keys, cross_values = project_kv(layer.second_sub_layer, encoder_states)
            hidden = hidden + attention(
                layer.second_sub_layer, layer.layer_norm_2(hidden), cross_keys, cross_values
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))
        return self.projection(self.decoder.final_layer_norm(hidden[:, -1:]))


class CachedDecoder(Decoder):
    """Core ML state stores self-attention keys and values between calls."""

    capacity = 544

    def __init__(self, model):
        super().__init__(model)
        for index, layer in enumerate(self.decoder.layers):
            attention_layer = layer.first_sub_layer
            shape = (1, attention_layer.h, self.capacity, attention_layer.d_k)
            self.register_buffer(f"keys_{index}", torch.zeros(shape, dtype=torch.float16))
            self.register_buffer(f"values_{index}", torch.zeros(shape, dtype=torch.float16))

    def forward(self, ids, encoder_states, positions, mask):
        embedding = self.decoder.embedding
        hidden = embedding.layer_norm(
            embedding.token_embedding(ids) + embedding.position_embedding(positions)
        )
        for index, layer in enumerate(self.decoder.layers):
            normed = layer.layer_norm_1(hidden)
            keys, values = project_kv(layer.first_sub_layer, normed)
            key_cache = getattr(self, f"keys_{index}")
            value_cache = getattr(self, f"values_{index}")
            indices = positions.reshape(1, 1, 1, 1).long().expand(1, layer.first_sub_layer.h, 1, layer.first_sub_layer.d_k)
            updated_keys = key_cache.float().scatter(2, indices, keys.float())
            updated_values = value_cache.float().scatter(2, indices, values.float())
            key_cache[:, :, :, :].copy_(updated_keys.to(torch.float16))
            value_cache[:, :, :, :].copy_(updated_values.to(torch.float16))
            hidden = hidden + attention(
                layer.first_sub_layer, normed, updated_keys.to(normed.dtype),
                updated_values.to(normed.dtype), mask,
            )
            cross_keys, cross_values = project_kv(layer.second_sub_layer, encoder_states)
            hidden = hidden + attention(
                layer.second_sub_layer, layer.layer_norm_2(hidden), cross_keys, cross_values
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))
        return self.projection(self.decoder.final_layer_norm(hidden[:, -1:]))


def export(component, module, examples, inputs, output, precision, directory, states=None):
    traced = torch.jit.trace(module.eval(), examples, check_trace=False)
    package = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=precision,
        inputs=inputs,
        outputs=[ct.TensorType(name=output, dtype=np.float32)],
        states=states,
        skip_model_load=True,
    )
    package.user_defined_metadata["source_model"] = "bodhan-ai/indic-transcribe-flex"
    package.user_defined_metadata["source_revision"] = REVISION
    package.save(str(directory / f"{component}.mlpackage"))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--component", choices=("frontend", "encoder", "decoder", "decoder_cached"), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.snapshot.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    if args.component == "frontend":
        from feature_extraction_indic_canary import IndicCanaryFeatureExtractor

        source = IndicCanaryFeatureExtractor.from_pretrained(str(args.snapshot), device="cpu")
        module = Frontend(source)
        generator = torch.Generator().manual_seed(73)
        for count in (16000, 17301, 480000):
            signal = torch.randn((1, count), generator=generator) * 0.1
            expected, _ = source(signal, torch.tensor([count]))
            torch.testing.assert_close(module(signal), expected, atol=3e-4, rtol=2e-4)
        export(
            "frontend", module, (signal[:, :16000],),
            [ct.TensorType(name="audio", shape=(1, ct.RangeDim(16000, 480000)))],
            "features", ct.precision.FLOAT32, args.output,
        )
        return

    from modeling_indic_canary import IndicCanaryForConditionalGeneration

    model = IndicCanaryForConditionalGeneration.from_pretrained(
        str(args.snapshot), dtype=torch.float32,
    ).eval()
    if args.component == "encoder":
        module = Encoder(model)
        export(
            "encoder", module, (torch.zeros(1, 128, 101),),
            [ct.TensorType(name="features", shape=(1, 128, ct.RangeDim(101, 3001)))],
            "encoder_states", ct.precision.FLOAT16, args.output,
        )
    elif args.component == "decoder":
        module = Decoder(model)
        ids = torch.tensor([[7, 4, 18, 89, 89, 5, 9, 11, 13, 15]], dtype=torch.int32)
        encoded = torch.zeros(1, 13, 1024)
        expected = model.lm_head(model.get_decoder()(ids, encoded, None)[:, -1:])
        positions = torch.arange(ids.size(1), dtype=torch.int32)[None, :]
        mask = (positions[:, :, None] < positions[:, None, :]).float()[:, None] * -10000.0
        examples = (ids, encoded, positions, mask)
        torch.testing.assert_close(module(*examples), expected, atol=5e-4, rtol=5e-4)
        length = ct.RangeDim(1, 1024, default=10)
        export(
            "decoder", module, examples,
            [
                ct.TensorType(name="ids", shape=(1, length), dtype=np.int32),
                ct.TensorType(name="encoder_states", shape=(1, ct.RangeDim(13, 376), 1024)),
                ct.TensorType(name="positions", shape=(1, length), dtype=np.int32),
                ct.TensorType(name="mask", shape=(1, 1, length, length)),
            ],
            "logits", ct.precision.FLOAT16, args.output,
        )

    else:
        module = CachedDecoder(model)
        encoded = torch.zeros(1, 13, 1024)
        ids = torch.tensor([[7, 4, 18, 89, 89, 5, 9, 11, 13, 15]], dtype=torch.int32)
        positions = torch.arange(ids.size(1), dtype=torch.int32)[None, :]
        cache_positions = torch.arange(module.capacity, dtype=torch.int32)
        mask = (positions[:, :, None] < cache_positions).float()[:, None] * -10000.0
        for position in range(ids.size(1)):
            examples = (
                ids[:, position:position + 1], encoded,
                positions[:, position:position + 1], mask[:, :, position:position + 1, :],
            )
            result = module(*examples)
        reference = model.lm_head(model.get_decoder()(ids, encoded, None)[:, -1:])
        torch.testing.assert_close(result, reference, atol=0.02, rtol=0.01)
        states = [
            ct.StateType(wrapped_type=ct.TensorType(shape=value.shape, dtype=np.float16), name=name)
            for name, value in module.named_buffers()
            if name.startswith(("keys_", "values_"))
        ]
        for name, value in module.named_buffers():
            if name.startswith(("keys_", "values_")):
                value.zero_()
        length = 1
        export(
            "decoder_cached", module, examples,
            [
                ct.TensorType(name="ids", shape=(1, length), dtype=np.int32),
                ct.TensorType(name="encoder_states", shape=(1, ct.RangeDim(13, 376), 1024)),
                ct.TensorType(name="positions", shape=(1, length), dtype=np.int32),
                ct.TensorType(name="mask", shape=(1, 1, length, module.capacity)),
            ],
            "logits", ct.precision.FLOAT16, args.output, states=states,
        )


if __name__ == "__main__":
    main()
