"""V3 FP32 frontend using the checkpoint's exact window/filterbank buffers.
DFT convolutions replace complex STFT for CoreML export; masks and unbiased
per-feature variance follow the pinned NeMo implementation. No quantization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Frontend(nn.Module):
    def __init__(self,window,filterbank,settings):
        super().__init__()
        if settings['exact_pad'] or settings['normalize']!='per_feature' or settings['log_zero_guard_type']!='add' or settings['mag_power']!=2.0:
            raise ValueError('source preprocessing contract not supported by this export')
        self.n_fft=settings['n_fft'];self.hop=settings['hop_length']
        self.preemphasis=settings['preemphasis'];self.log_guard=settings['log_guard'];self.epsilon=settings['normalization_epsilon']
        padded=torch.zeros(self.n_fft,dtype=torch.float64)
        start=(self.n_fft-settings['win_length'])//2
        padded[start:start+settings['win_length']]=torch.as_tensor(window,dtype=torch.float64)
        n=torch.arange(self.n_fft,dtype=torch.float64)
        k=torch.arange(self.n_fft//2+1,dtype=torch.float64)
        phase=2*math.pi*k[:,None]*n[None,:]/self.n_fft
        self.register_buffer('real_basis',(torch.cos(phase)*padded).float()[:,None,:])
        self.register_buffer('imag_basis',(-torch.sin(phase)*padded).float()[:,None,:])
        self.register_buffer('filterbank',torch.as_tensor(filterbank,dtype=torch.float32).reshape(128,self.n_fft//2+1))

    def forward(self,audio_signal,audio_length):
        positions=torch.arange(audio_signal.shape[-1],device=audio_signal.device)[None,:]
        valid_audio=positions<audio_length[:,None]
        previous=F.pad(audio_signal[:,:-1],(1,0))
        emphasized=audio_signal-self.preemphasis*previous
        emphasized=torch.where(valid_audio,emphasized,torch.zeros_like(emphasized))
        padded=F.pad(emphasized[:,None,:],(self.n_fft//2,self.n_fft//2))
        real=F.conv1d(padded,self.real_basis,stride=self.hop)
        imag=F.conv1d(padded,self.imag_basis,stride=self.hop)
        # Preserve source magnitude -> power arithmetic before filterbank/log.
        magnitude=torch.sqrt(real.square()+imag.square())
        power=magnitude.square()
        logged=torch.log(torch.matmul(self.filterbank,power)+self.log_guard)
        valid_length=audio_length//self.hop
        mask=torch.arange(logged.shape[-1],device=logged.device)[None,None,:]<valid_length[:,None,None]
        denom=torch.clamp(valid_length,min=1).to(logged.dtype)[:,None,None]
        mean=torch.where(mask,logged,torch.zeros_like(logged)).sum(-1,keepdim=True)/denom
        centered=torch.where(mask,logged-mean,torch.zeros_like(logged))
        var=centered.square().sum(-1,keepdim=True)/torch.clamp(denom-1,min=1)
        normalized=(logged-mean)/(torch.sqrt(var)+self.epsilon)
        mel=torch.where(mask,normalized,torch.zeros_like(normalized))
        return mel,valid_length.to(torch.int32)
