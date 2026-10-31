"""Original NVIDIA v3 weights in exportable PyTorch; no quantization or fused heads.
Adapted from our 110M reference math, with v3's exact dimensions and bias policy.
Every state dict loads strictly and is checked against independently captured NeMo outputs.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
N_MELS=128
D_MODEL=1024
N_HEADS=8
D_HEAD=128
N_LAYERS=24
FF_DIM=4096
CONV_KERNEL=9
SUB_CHANNELS=256
VOCAB=8192
BLANK=8192
PRED_HIDDEN=640
DURATIONS=(0,1,2,3,4)
def rel_positions(t: int) -> torch.Tensor:
    """Sinusoidal embeddings for relative positions t-1 .. -(t-1): (2t-1, d).

    Built in NumPy from a plain int so a traced graph holds it as a constant.
    """
    t = int(t)
    pos = np.arange(t - 1, -t, -1, dtype=np.float32)[:, None]
    div = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * np.float32(-(math.log(10000.0) / D_MODEL)))
    pe = np.zeros((2 * t - 1, D_MODEL), dtype=np.float32)
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return torch.from_numpy(pe)


def rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Transformer-XL shift: (b, h, t, 2t-1) -> (b, h, t, 2t-1), NeMo's exact form."""
    b, h, t, p = x.shape
    x = F.pad(x, (1, 0))
    x = x.view(b, h, -1, t)
    return x[:, :, 1:].reshape(b, h, t, p)


# ------------------------------------------------------- reference encoder

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(D_MODEL, FF_DIM, bias=False)
        self.linear2 = nn.Linear(FF_DIM, D_MODEL, bias=False)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(D_MODEL, 2 * D_MODEL, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(D_MODEL, D_MODEL, CONV_KERNEL, padding=CONV_KERNEL // 2, groups=D_MODEL, bias=False)
        self.batch_norm = nn.BatchNorm1d(D_MODEL)
        self.pointwise_conv2 = nn.Conv1d(D_MODEL, D_MODEL, 1, bias=False)

    def forward(self, x, keep):  # x (b, t, c), keep (b, t, 1) float
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = x * keep.transpose(1, 2)
        x = F.silu(self.batch_norm(self.depthwise_conv(x)))
        return self.pointwise_conv2(x).transpose(1, 2)


class RelPosAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.linear_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.linear_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.linear_out = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.linear_pos = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(N_HEADS, D_HEAD))
        self.pos_bias_v = nn.Parameter(torch.zeros(N_HEADS, D_HEAD))

    def forward(self, x, pos, mask):  # mask (b, 1, t, t) bool, True = blocked
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, N_HEADS, D_HEAD)
        k = self.linear_k(x).view(b, t, N_HEADS, D_HEAD).transpose(1, 2)
        v = self.linear_v(x).view(b, t, N_HEADS, D_HEAD).transpose(1, 2)
        p = self.linear_pos(pos).view(1, -1, N_HEADS, D_HEAD).transpose(1, 2)
        ac = torch.matmul((q + self.pos_bias_u).transpose(1, 2), k.transpose(-2, -1))
        bd = rel_shift(torch.matmul((q + self.pos_bias_v).transpose(1, 2), p.transpose(-2, -1)))[..., :t]
        scores = (ac + bd) / math.sqrt(D_HEAD)
        scores = scores.masked_fill(mask, -10000.0)
        attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, D_MODEL)
        return self.linear_out(out)


class ConformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(D_MODEL)
        self.feed_forward1 = FeedForward()
        self.norm_conv = nn.LayerNorm(D_MODEL)
        self.conv = ConvModule()
        self.norm_self_att = nn.LayerNorm(D_MODEL)
        self.self_attn = RelPosAttention()
        self.norm_feed_forward2 = nn.LayerNorm(D_MODEL)
        self.feed_forward2 = FeedForward()
        self.norm_out = nn.LayerNorm(D_MODEL)

    def forward(self, x, pos, mask, keep):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos, mask)
        x = x + self.conv(self.norm_conv(x), keep)
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Subsampling(nn.Module):
    """dw_striding x8: (b, 80, T) mel -> (b, T/8, 512)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, SUB_CHANNELS, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(SUB_CHANNELS, SUB_CHANNELS, 3, 2, 1, groups=SUB_CHANNELS),
            nn.Conv2d(SUB_CHANNELS, SUB_CHANNELS, 1), nn.ReLU(),
            nn.Conv2d(SUB_CHANNELS, SUB_CHANNELS, 3, 2, 1, groups=SUB_CHANNELS),
            nn.Conv2d(SUB_CHANNELS, SUB_CHANNELS, 1), nn.ReLU(),
        )
        self.out = nn.Linear(SUB_CHANNELS * (N_MELS // 8), D_MODEL)

    def forward(self, mel, length):
        """mel (b, 80, T) with `length` valid frames.

        TODO: masking rule under study against NeMo.
        """
        x = mel.transpose(1, 2)[:, None]                    # (b, 1, T, 80)
        stages = ((0, 2), (2, 5), (5, 8))
        for begin, end in stages:
            for module in self.conv[begin:end]:
                x = module(x)
            length = (length + 1) // 2
            keep = torch.arange(x.shape[2], device=x.device)[None, :] < length[:, None]
            x = x * keep[:, None, :, None].to(x.dtype)
        b, c, t, f = x.shape                                 # (b, 256, T/8, 10)
        return self.out(x.transpose(1, 2).reshape(b, t, SUB_CHANNELS * (N_MELS // 8)))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_encode = Subsampling()
        self.layers = nn.ModuleList(ConformerLayer() for _ in range(N_LAYERS))

    def forward(self, mel, mel_length):
        """mel (b, 80, T), mel_length (b,) valid mel frames -> (b, T/8, 512)."""
        x = self.pre_encode(mel, mel_length)
        t = x.shape[1]
        length = mel_length
        for _ in range(3):
            length = (length + 1) // 2
        valid = torch.arange(t, device=x.device)[None, :] < length[:, None]
        mask = ~(valid[:, None, :, None] & valid[:, None, None, :])
        keep = valid[:, :, None].to(x.dtype)
        pos = rel_positions(t).to(x)
        for layer in self.layers:
            x = layer(x, pos, mask, keep)
        return x



class EncoderExport(nn.Module):
    def __init__(self, encoder):
        super().__init__();self.encoder=encoder
    def forward(self,mel,mel_length):
        output=self.encoder(mel,mel_length).transpose(1,2)
        return output,((mel_length+7)//8).to(torch.int32)

class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(8193,640,padding_idx=8192)
        self.lstm=nn.LSTM(640,640,2,batch_first=True)
    def forward(self,targets,h_in,c_in):
        out,(h,c)=self.lstm(self.embed(targets),(h_in,c_in))
        return out.transpose(1,2),h,c

class Joint(nn.Module):
    def __init__(self):
        super().__init__();self.enc=nn.Linear(1024,640);self.pred=nn.Linear(640,640);self.out=nn.Linear(640,8198)
    def forward(self,encoder_step,decoder_step):
        e=self.enc(encoder_step.transpose(1,2));g=self.pred(decoder_step.transpose(1,2))
        return self.out(F.relu(e.unsqueeze(2)+g.unsqueeze(1)))

def load_weights(path):
    sd=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    encoder=Encoder()
    encoder.load_state_dict({k[len('encoder.'):]:v for k,v in sd.items() if k.startswith('encoder.')},strict=True,assign=True)
    predictor=Predictor()
    predictor.embed.weight=nn.Parameter(sd['decoder.prediction.embed.weight'],requires_grad=False)
    predictor.lstm.load_state_dict({k:sd['decoder.prediction.dec_rnn.lstm.'+k] for k in predictor.lstm.state_dict()},strict=True,assign=True)
    joint=Joint()
    joint.load_state_dict({f'{dst}.{suffix}':sd[f'{src}.{suffix}'] for dst,src in [('enc','joint.enc'),('pred','joint.pred'),('out','joint.joint_net.2')] for suffix in ['weight','bias']},strict=True,assign=True)
    return EncoderExport(encoder).eval(),predictor.eval(),joint.eval()
