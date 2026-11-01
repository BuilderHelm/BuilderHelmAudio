"""Combine pinned 15/30-second FP16 exports with shared weights for the Swift runtime."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
import coremltools as ct
from export import CHECKPOINT_SHA256,SOURCE_REVISION

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--short',type=Path,required=True);p.add_argument('--long',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--split-encoder',action='store_true',help='Partition each encoder into two sequential programs and share constants across window sizes');a=p.parse_args()
 if a.out.exists():p.error('--out must be a new directory')
 config=json.loads((a.short/'config.json').read_text());record=json.loads((a.long/'export-record.json').read_text())
 if config['source_revision']!=SOURCE_REVISION or config['precision']!='fp16' or config['window_samples']!=240000:raise ValueError('Expected pinned FP16 15-second export')
 if record['source_checkpoint_sha256']!=CHECKPOINT_SHA256 or record['precision']!='fp16' or record['window_seconds']!=30:raise ValueError('Expected pinned FP16 30-second export')
 for path,h in config['files'].items():
  if sha(a.short/path)!=h:raise ValueError('Short export changed: '+path)
 for path,h in record['files'].items():
  if sha(a.long/path)!=h:raise ValueError('Long export changed: '+path)
 a.out.mkdir(parents=True)
 for name in (['Preprocessor'] if a.split_encoder else ['Preprocessor','Encoder']):
  d=ct.utils.MultiFunctionDescriptor();d.add_function(str(a.short/(name+'.mlpackage')),'main','w15');d.add_function(str(a.long/(name+'.mlpackage')),'main','w30');d.default_function_name='w15';ct.utils.save_multifunction(d,str(a.out/(name+'.mlpackage')))
 if a.split_encoder:
  with tempfile.TemporaryDirectory(prefix='parakeet-v3-split-') as temporary:
   temporary=Path(temporary)
   for source,label in [(a.short,'short'),(a.long,'long')]:
    ct.models.utils.bisect_model(str(source/'Encoder.mlpackage'),str(temporary/label),check_output_correctness=False)
   for part,name in [(1,'Encoder'),(2,'EncoderTail')]:
    d=ct.utils.MultiFunctionDescriptor()
    for label,function in [('short','w15'),('long','w30')]:d.add_function(str(temporary/label/f'Encoder_chunk{part}.mlpackage'),'main',function)
    d.default_function_name='w15';ct.utils.save_multifunction(d,str(a.out/(name+'.mlpackage')))
  config['encoder_stages']=['Encoder','EncoderTail']
 for name in ['Predictor.mlpackage','Joint.mlpackage']:shutil.copytree(a.short/name,a.out/name)
 shutil.copy2(a.short/'vocab.json',a.out/'vocab.json')
 config['windows']=[dict(samples=240000,function='w15'),dict(samples=480000,function='w30')];config.pop('experimental_long_audio',None)
 config['long_audio']=dict(policy='quiet-pause with right context, frame-aligned overlap fallback, bounded blank recovery',long_window_samples=480000,short_window_samples=240000)
 config['files']={str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file()}
 (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
 (a.out/'merge-record.json').write_text(json.dumps(dict(source_revision=SOURCE_REVISION,source_checkpoint_sha256=CHECKPOINT_SHA256,short_config_sha256=sha(a.short/'config.json'),long_export_record_sha256=sha(a.long/'export-record.json'),coremltools=ct.__version__,constant_deduplication=True,encoder_stages=2 if a.split_encoder else 1,partition_correctness='Requires corpus validation; random utility check disabled for large-model memory use' if a.split_encoder else 'not partitioned'),indent=2)+'\n')
if __name__=='__main__':main()
