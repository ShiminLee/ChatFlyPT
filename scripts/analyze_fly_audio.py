"""Analyze an external mono WAV without publishing it as simulated fly speech."""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chatflypt.acoustic_readout import AcousticReadout
from chatflypt.paths import dump,sha256


def main():
    from scipy.io import wavfile
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav',type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();rate,samples=wavfile.read(args.wav)
    if samples.dtype.kind=='i':samples=samples.astype(np.float32)/max(abs(np.iinfo(samples.dtype).min),np.iinfo(samples.dtype).max)
    elif samples.dtype==np.uint8:samples=(samples.astype(np.float32)-128)/128
    elif samples.dtype.kind!='f':raise ValueError('Unsupported WAV encoding')
    result=AcousticReadout().analyze(samples,rate,source='external_wav')
    result['recording_sha256']=sha256(args.wav);dump(args.output,result)
    print(f"Acoustic intervals: {len(result['intervals'])}; semantics decoded: false; report: {args.output}")


if __name__=='__main__':main()
