"""Combine pinned Unified FP16 exports into shared-weight window functions for the Swift runtime.

The 15-second export carries every part; the 30-second and any `--extra` windows (5, 10, 20 s)
carry frontend and encoder. Each window becomes function `w<seconds>` with Core ML constant
deduplication; `w15` is the default function.
"""
import argparse,hashlib,json,shutil,tempfile
from pathlib import Path
import coremltools as ct
from export import CHECKPOINT_SHA256,SOURCE_REVISION

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--short',type=Path,required=True);p.add_argument('--long',type=Path,required=True)
 p.add_argument('--extra',type=Path,nargs='*',default=[],help='additional frontend+encoder exports (5, 10 or 20 s)')
 p.add_argument('--out',type=Path,required=True);p.add_argument('--split-encoder',action='store_true',help='Partition each encoder into two sequential programs and share constants across window sizes');a=p.parse_args()
 if a.out.exists():p.error('--out must be a new directory')
 config=json.loads((a.short/'config.json').read_text())
 if config['architecture']!='parakeet-unified-en-0.6b' or config['source_revision']!=SOURCE_REVISION or config['precision']!='fp16' or config['window_samples']!=240000:raise ValueError('Expected pinned FP16 15-second Unified export')
 for path,h in config['files'].items():
  if sha(a.short/path)!=h:raise ValueError('Short export changed: '+path)
 windows={15:a.short}
 for folder in [a.long,*a.extra]:
  record=json.loads((folder/'export-record.json').read_text())
  seconds=record['window_seconds']
  if record['source_checkpoint_sha256']!=CHECKPOINT_SHA256 or record['precision']!='fp16' or record.get('mode')!='offline' or seconds in windows or (folder==a.long)!=(seconds==30):
   raise ValueError(f'Expected pinned FP16 Unified export with a distinct window: {folder}')
  for path,h in record['files'].items():
   if sha(folder/path)!=h:raise ValueError(f'{seconds}-second export changed: '+path)
  windows[seconds]=folder
 windows=dict(sorted(windows.items()))
 a.out.mkdir(parents=True)
 def merge(name,source):
  d=ct.utils.MultiFunctionDescriptor()
  for seconds in windows:d.add_function(str(source(seconds)),'main',f'w{seconds}')
  d.default_function_name='w15';ct.utils.save_multifunction(d,str(a.out/(name+'.mlpackage')))
 merge('Preprocessor',lambda s:windows[s]/'Preprocessor.mlpackage')
 if a.split_encoder:
  with tempfile.TemporaryDirectory(prefix='parakeet-unified-split-') as temporary:
   temporary=Path(temporary)
   for seconds,folder in windows.items():
    ct.models.utils.bisect_model(str(folder/'Encoder.mlpackage'),str(temporary/f'w{seconds}'),check_output_correctness=False)
   for part,name in [(1,'Encoder'),(2,'EncoderTail')]:merge(name,lambda s:temporary/f'w{s}'/f'Encoder_chunk{part}.mlpackage')
  config['encoder_stages']=['Encoder','EncoderTail']
 else:merge('Encoder',lambda s:windows[s]/'Encoder.mlpackage')
 for name in ['Predictor.mlpackage','Joint.mlpackage']:shutil.copytree(a.short/name,a.out/name)
 for name in ['vocab.json','RnntDecoder.bin']:shutil.copy2(a.short/name,a.out/name)
 config['windows']=[dict(samples=s*16000,function=f'w{s}') for s in windows];config.pop('experimental_long_audio',None)
 config['long_audio']=dict(policy='30-second windows; quiet-pause with right context, frame-aligned overlap fallback, bounded blank recovery',long_window_samples=480000,short_window_samples=240000)
 config['export_metadata']=dict(constant_deduplication=True,encoder_partition='Two approximately equal-weight sequential ML Programs; FP16 unchanged; corresponding window chunks share constants' if a.split_encoder else 'not partitioned')
 config['files']={str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file()}
 (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 (a.out/'merge-record.json').write_text(json.dumps(dict(source_revision=SOURCE_REVISION,source_checkpoint_sha256=CHECKPOINT_SHA256,short_config_sha256=sha(a.short/'config.json'),
  window_export_records={f'w{s}':sha(f/'export-record.json') for s,f in windows.items() if s!=15},coremltools=ct.__version__,constant_deduplication=True,encoder_stages=2 if a.split_encoder else 1,
  partition_correctness='Requires corpus validation; random utility check disabled for large-model memory use' if a.split_encoder else 'not partitioned'),indent=2)+'\n')
if __name__=='__main__':main()
