"""Fetch pinned public DAS assets and reject changed content before publishing."""
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse

ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/janclemenslab/das-menagerie/releases/download/Dmel_single_v1.0/'
FILES={
    'pulse+sine_model.h5':'ca03ced978a8c5f5e294ba1cfb7e3d24cc084a8f4d8bdede938aae6608caef32',
    'pulse+sine_params.yaml':'1c7d220c6c622ad7887e6684c91a78490a4f3e3caf34d949340980846d512065',
    'data.npz':'21c4455b8baec37ec42dcf166ba9454fa7e20764492a74e65fead46a612a2df9',
}


def main():
    folder=ROOT/'research/outward-signals/das-pretrained';folder.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,expected in FILES.items():
        path=folder/name;url=BASE+urllib.parse.quote(name)
        data=path.read_bytes() if path.exists() else urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('DAS asset hash mismatch: '+name)
        if not path.exists():path.write_bytes(data)
        manifest.append({'url':url,'path':name,'sha256':expected,'bytes':len(data)})
        print(name,expected,flush=True)
    (folder/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
