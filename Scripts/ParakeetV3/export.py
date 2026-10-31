#!/usr/bin/env python3
"""Export a checksum-pinned original NVIDIA v3 checkpoint to local CoreML packages.
No NeMo runtime, competitor code/weights, quantization, CTC or top-K truncation.
"""
import argparse,gc,hashlib,json,math,re,shutil,tarfile,tempfile,unicodedata,wave
from pathlib import Path
import coremltools as ct
import numpy as np
import torch
import yaml
from frontend import Frontend
from model import load_weights
CHECKPOINT_SHA256='3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'
SOURCE_REVISION='541d1f99c6b0c3cd0b11a95167540bb8edefd82b'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--nemo',required=True,type=Path);p.add_argument('--example-wav',required=True,type=Path)
 p.add_argument('--out',required=True,type=Path);p.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
 p.add_argument('--window-seconds',type=int,choices=[15,30],default=15)
 p.add_argument('--parts',nargs='+',choices=['frontend','encoder','predictor','joint'],default=['frontend','encoder','predictor','joint'])
 args=p.parse_args()
 if args.out.exists():p.error('--out must be a new directory; existing exports are never overwritten')
 if sha(args.nemo)!=CHECKPOINT_SHA256:raise ValueError('Expected pinned original NVIDIA parakeet-tdt-0.6b-v3 checkpoint')
 with wave.open(str(args.example_wav),'rb') as wav:
  if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getcomptype())!=(16000,1,2,'NONE'):raise ValueError('Example must be mono PCM16 WAV at 16 kHz')
  if not 8000<=wav.getnframes()<=args.window_seconds*16000:raise ValueError('Provide real speech within the selected window')
  samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
 torch.manual_seed(0);torch.set_num_threads(4)
 args.out.mkdir(parents=True)
 with tempfile.TemporaryDirectory(prefix='bha-v3-weights-') as folder,tarfile.open(args.nemo) as archive:
  config=yaml.safe_load(archive.extractfile('./model_config.yaml'))
  if config['encoder']['d_model']!=1024 or config['encoder']['n_layers']!=24 or config['decoder']['prednet']['pred_rnn_layers']!=2:
   raise ValueError('Unsupported source architecture')
  pieces=config['joint']['vocabulary']
  if len(pieces)!=8192 or config['decoding']['durations']!=[0,1,2,3,4]:raise ValueError('Invalid vocabulary/duration contract')
  weights=Path(folder)/'weights.ckpt'
  with archive.extractfile('./model_weights.ckpt') as f,weights.open('wb') as out:shutil.copyfileobj(f,out,8*1024*1024)
  state=torch.load(weights,map_location='cpu',mmap=True,weights_only=True)
  settings=dict(n_fft=512,hop_length=160,win_length=400,preemphasis=.97,mag_power=2.0,normalize='per_feature',exact_pad=False,
   log_zero_guard_type='add',log_guard=2.0**-24,normalization_epsilon=1e-5)
  pre=config['preprocessor']
  if pre['features']!=128 or pre['sample_rate']!=16000 or pre['n_fft']!=512 or pre['normalize']!='per_feature':raise ValueError('Unexpected source frontend')
  frontend=Frontend(state['preprocessor.featurizer.window'].numpy(),state['preprocessor.featurizer.fb'].numpy(),settings).eval()
  encoder,predictor,joint=load_weights(weights)
  audio=torch.zeros(1,args.window_seconds*16000);audio[0,:len(samples)]=torch.from_numpy(samples)
  length=torch.tensor([len(samples)],dtype=torch.int32);h=torch.zeros(2,1,640);c=torch.zeros_like(h)
  target=torch.tensor([[8192]],dtype=torch.int32)
  with torch.no_grad():
   mel,ml=frontend(audio,length);enc,el=encoder(mel,ml);g,h2,c2=predictor(target,h,c);logits=joint(enc[:,:,:1],g)
  np.savez_compressed(args.out/'example-reference.npz',audio=audio.numpy(),audio_length=length.numpy(),mel=mel.numpy(),mel_length=ml.numpy(),encoder=enc.numpy(),encoder_length=el.numpy(),decoder=g.numpy(),h_out=h2.numpy(),c_out=c2.numpy(),logits=logits.numpy())
  contracts={'frontend':('Preprocessor',frontend,(audio,length),['audio_signal','audio_length'],['mel','mel_length']),
   'encoder':('Encoder',encoder,(mel,ml),['mel','mel_length'],['encoder','encoder_length']),
   'predictor':('Predictor',predictor,(target,h,c),['targets','h_in','c_in'],['decoder','h_out','c_out']),
   'joint':('Joint',joint,(enc[:,:,:1].contiguous(),g),['encoder_step','decoder_step'],['logits'])}
  for part in args.parts:
   name,model,example,inputs,outputs=contracts[part]
   with torch.no_grad():traced=torch.jit.trace(model,example,strict=True,check_trace=False)
   precision=ct.precision.FLOAT32 if args.precision=='fp32' or part=='frontend' else ct.precision.FLOAT16
   converted=ct.convert(traced,convert_to='mlprogram',minimum_deployment_target=ct.target.iOS18,compute_precision=precision,
    compute_units=ct.ComputeUnit.CPU_ONLY,skip_model_load=True,
    inputs=[ct.TensorType(name=n,shape=tuple(t.shape),dtype=np.int32 if t.dtype==torch.int32 else np.float32) for n,t in zip(inputs,example)],
    outputs=[ct.TensorType(name=n,dtype=np.int32 if n.endswith('length') else np.float32) for n in outputs])
   if precision==ct.precision.FLOAT32 and 'fp16' in str(converted._mil_program):raise ValueError('FP16 dtype in FP32 component')
   converted.author='BuilderHelmAudio; original source weights NVIDIA'
   converted.license='NVIDIA parakeet-tdt-0.6b-v3, CC BY 4.0. https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3'
   converted.short_description=f'Unquantized Parakeet v3 {name}; {args.precision}, frontend/interfaces FP32'
   converted.save(str(args.out/(name+'.mlpackage')))
   del converted,traced;gc.collect()
  write(args.out/'vocab.json',pieces)
  punctuation=sorted({c for piece in pieces if not re.match(r'^\[.*\]$|^<.*>$|^##|^▁|^\s*$',piece) for c in piece if unicodedata.category(c).startswith('P')})
  files={str(f.relative_to(args.out)):sha(f) for f in args.out.rglob('*') if f.is_file()}
  write(args.out/'export-record.json',dict(source_repository='nvidia/parakeet-tdt-0.6b-v3',source_revision=SOURCE_REVISION,source_checkpoint_sha256=CHECKPOINT_SHA256,
   precision=args.precision,window_seconds=args.window_seconds,parts=args.parts,torch=torch.__version__,coremltools=ct.__version__,example_wav_sha256=sha(args.example_wav),files=files,
   validation='Example Torch reference only; full independent NeMo/CoreML parity must be run before accepting a new exporter revision.'))
  if set(args.parts)==set(contracts):
   write(args.out/'config.json',dict(architecture='parakeet-tdt-0.6b-v3',precision=args.precision,sample_rate=16000,window_samples=args.window_seconds*16000,blank_id=8192,vocabulary_size=8192,
    durations=[0,1,2,3,4],max_symbols=10,punctuation=punctuation,files={k:v for k,v in files.items() if k.endswith('vocab.json') or '.mlpackage/' in k},
    source_revision=SOURCE_REVISION,frontend_precision='fp32',license='NVIDIA CC BY 4.0',experimental_long_audio=False))
 print(args.out)
if __name__=='__main__':main()
