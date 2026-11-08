"""Build the INT8 encoder variant of a staged Unified release folder.

Each window's FP16 encoder is compressed with weight-only, per-output-channel, symmetric
linear INT8 (compute stays FP16), then the 15/30-second functions are merged with shared
constants. The encoder is not partitioned: it is below 1 GB. Frontend, predictor, joint
and vocabulary are the release folder's own files, so the variant config lists their
root checksums and a downloaded variant folder is self-contained.
"""
import argparse,hashlib,json,tempfile
from pathlib import Path
import coremltools as ct
import coremltools.optimize.coreml as cto
from export import CHECKPOINT_SHA256,SOURCE_REVISION
SHARED=('Preprocessor.mlpackage','Predictor.mlpackage','Joint.mlpackage','vocab.json')
SCHEME='int8-linear-symmetric-per-channel'

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--short',type=Path,required=True);p.add_argument('--long',type=Path,required=True)
 p.add_argument('--release',type=Path,required=True,help='staged FP16 release folder');p.add_argument('--out',type=Path,required=True,help='new folder, usually <release>/variants/int8');a=p.parse_args()
 if a.out.exists():p.error('--out must be a new directory')
 config=json.loads((a.release/'config.json').read_text())
 if config['architecture']!='parakeet-unified-en-0.6b' or config['source_revision']!=SOURCE_REVISION or config['precision']!='fp16' or not config.get('staged_preprocessor'):raise ValueError('Expected a staged FP16 Unified release folder')
 for path,h in config['files'].items():
  if sha(a.release/path)!=h:raise ValueError('Release folder changed: '+path)
 sources={}
 for folder,seconds,manifest in [(a.short,15,'config.json'),(a.long,30,'export-record.json')]:
  record=json.loads((folder/'export-record.json').read_text())
  if record['source_checkpoint_sha256']!=CHECKPOINT_SHA256 or record['precision']!='fp16' or record['window_seconds']!=seconds:raise ValueError(f'Expected pinned FP16 {seconds}-second export')
  for path,h in record['files'].items():
   if path.startswith('Encoder.mlpackage/') and sha(folder/path)!=h:raise ValueError(f'{seconds}-second encoder changed: '+path)
  sources[f'w{seconds}']={k:v for k,v in record['files'].items() if k.startswith('Encoder.mlpackage/')}
 a.out.mkdir(parents=True)
 quantizer=cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode='linear_symmetric',dtype='int8',granularity='per_channel'))
 with tempfile.TemporaryDirectory(prefix='parakeet-unified-int8-') as temporary:
  temporary=Path(temporary);d=ct.utils.MultiFunctionDescriptor()
  for folder,function in [(a.short,'w15'),(a.long,'w30')]:
   model=cto.linear_quantize_weights(ct.models.MLModel(str(folder/'Encoder.mlpackage'),skip_model_load=True),quantizer)
   model.short_description=f'Parakeet Unified EN encoder ({function}); {SCHEME} weights, FP16 compute'
   model.save(str(temporary/f'{function}.mlpackage'));d.add_function(str(temporary/f'{function}.mlpackage'),'main',function)
  d.default_function_name='w15';ct.utils.save_multifunction(d,str(a.out/'Encoder.mlpackage'))
 files={k:v for k,v in config['files'].items() if k.startswith(SHARED)}
 files.update({str(f.relative_to(a.out)):sha(f) for f in a.out.rglob('*') if f.is_file()})
 variant=dict(config);variant.pop('encoder_stages',None)
 variant.update(variant='int8',encoder_weights=SCHEME,files=files,export_metadata=dict(constant_deduplication=True,encoder_partition='not partitioned (below 1 GB)',encoder_weights=SCHEME))
 (a.out/'config.json').write_text(json.dumps(variant,indent=2)+'\n')
 (a.out/'variant-record.json').write_text(json.dumps(dict(source_revision=SOURCE_REVISION,source_checkpoint_sha256=CHECKPOINT_SHA256,release_config_sha256=sha(a.release/'config.json'),
  source_fp16_encoders=sources,scheme=SCHEME,coremltools=ct.__version__,shared_files=list(SHARED)),indent=2)+'\n')
 print(a.out)
if __name__=='__main__':main()
