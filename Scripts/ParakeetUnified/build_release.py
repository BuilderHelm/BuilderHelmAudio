#!/usr/bin/env python3
"""Build the complete Unified release folder from the pinned NVIDIA checkpoint.

  <out>/work/short   15 s FP16 export (all parts)
  <out>/work/long    30 s FP16 export (frontend, encoder)
  <out>/work/w5 …    5, 10 and 20 s FP16 exports (frontend, encoder)
  <out>/work/merged  5/10/15/20/30 s multifunction, FP16 encoder split in two programs
  <out>/release      staged frontend; FP16 at the root, INT8 under variants/int8

Every step refuses existing outputs and checks its inputs' checksums.
"""
import argparse,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
# Smaller windows cut Neural Engine time for shorter requests (INT8 encoder on M5: 5 s 9.5 ms,
# 10 s 14 ms, 15 s 26 ms, 20 s 35 ms, 30 s 82 ms); the weights are shared, so downloads don't grow.
EXTRA_WINDOWS=(5,10,20)

def run(*args):
 print('+',' '.join(map(str,args)),flush=True);subprocess.run([sys.executable,*map(str,args)],check=True,cwd=here)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nemo',type=Path,required=True);p.add_argument('--example-wav',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():p.error('--out must be a new directory')
 work=a.out/'work';release=a.out/'release'
 run(here/'export.py','--nemo',a.nemo,'--example-wav',a.example_wav,'--precision','fp16','--out',work/'short')
 run(here/'export.py','--nemo',a.nemo,'--example-wav',a.example_wav,'--precision','fp16','--window-seconds',30,'--parts','frontend','encoder','--out',work/'long')
 extra=[work/f'w{s}' for s in EXTRA_WINDOWS]
 for s,folder in zip(EXTRA_WINDOWS,extra):
  run(here/'export.py','--nemo',a.nemo,'--example-wav',a.example_wav,'--precision','fp16','--window-seconds',s,'--parts','frontend','encoder','--out',folder)
 run(here/'merge_windows.py','--short',work/'short','--long',work/'long','--extra',*extra,'--out',work/'merged','--split-encoder')
 run(here/'stage_preprocessor.py',work/'merged',release)
 run(here/'quantize_int8.py','--short',work/'short','--long',work/'long','--extra',*extra,'--release',release,'--out',release/'variants'/'int8')
 print(release)
if __name__=='__main__':main()
