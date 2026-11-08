#!/usr/bin/env python3
"""Build the complete Unified release folder from the pinned NVIDIA checkpoint.

  <out>/work/short   15 s FP16 export (all parts)
  <out>/work/long    30 s FP16 export (frontend, encoder)
  <out>/work/merged  15/30 s multifunction, FP16 encoder split in two programs
  <out>/release      staged frontend; FP16 at the root, INT8 under variants/int8

Every step refuses existing outputs and checks its inputs' checksums.
"""
import argparse,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent

def run(*args):
 print('+',' '.join(map(str,args)),flush=True);subprocess.run([sys.executable,*map(str,args)],check=True,cwd=here)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--nemo',type=Path,required=True);p.add_argument('--example-wav',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():p.error('--out must be a new directory')
 work=a.out/'work';release=a.out/'release'
 run(here/'export.py','--nemo',a.nemo,'--example-wav',a.example_wav,'--precision','fp16','--out',work/'short')
 run(here/'export.py','--nemo',a.nemo,'--example-wav',a.example_wav,'--precision','fp16','--window-seconds',30,'--parts','frontend','encoder','--out',work/'long')
 run(here/'merge_windows.py','--short',work/'short','--long',work/'long','--out',work/'merged','--split-encoder')
 run(here/'stage_preprocessor.py',work/'merged',release)
 run(here/'quantize_int8.py','--short',work/'short','--long',work/'long','--release',release,'--out',release/'variants'/'int8')
 print(release)
if __name__=='__main__':main()
