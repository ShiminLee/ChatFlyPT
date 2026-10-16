"""Download hash-pinned official local-model artifacts into the isolated report tree."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'reports/local-model-2026-09-19'
REVISION = '90862c4b9d2787eaed51d12237eafdfe7c5f6077'
ARTIFACTS = [
    {'name': 'Qwen3-1.7B-Q8_0.gguf', 'kind': 'model', 'bytes': 1834426016,
     'sha256': '061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a',
     'url': f'https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/{REVISION}/Qwen3-1.7B-Q8_0.gguf?download=true'},
    {'name': 'llama-b10964-bin-win-cuda-12.4-x64.zip', 'kind': 'runner', 'bytes': 254067651,
     'sha256': '264f20d7ee3860aecca9ec12418357a9f3e80349a2b186f66c63859ded1a9593',
     'url': 'https://github.com/ggml-org/llama.cpp/releases/download/b10964/llama-b10964-bin-win-cuda-12.4-x64.zip'},
    {'name': 'cudart-llama-bin-win-cuda-12.4-x64.zip', 'kind': 'runner', 'bytes': 391443627,
     'sha256': '8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6',
     'url': 'https://github.com/ggml-org/llama.cpp/releases/download/b10964/cudart-llama-bin-win-cuda-12.4-x64.zip'},
]
MODEL_4B_REVISION = 'e6f794d44f9395d0184a966c27b5ae99ea356fcb'
MODEL_4B = {'name': 'qwen3-4b-instruct-2507-q8_0.gguf', 'kind': 'model', 'bytes': 4280403520,
    'sha256': 'ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1',
    'url': f'https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF/resolve/{MODEL_4B_REVISION}/qwen3-4b-instruct-2507-q8_0.gguf?download=true'}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download(spec):
    target = BASE / ('models' if spec['kind'] == 'model' else 'downloads') / spec['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != spec['bytes'] or digest(target) != spec['sha256']:
        part = target.with_suffix(target.suffix+'.part')
        start = time.perf_counter()
        for attempt in range(3):
            try:
                offset = part.stat().st_size if part.exists() else 0
                request = urllib.request.Request(spec['url'], headers={'User-Agent': 'ChatFlyPT-local-evaluation/1.0',
                    **({'Range': f'bytes={offset}-'} if offset else {})})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206:
                        offset = 0
                    with part.open('ab' if offset else 'wb') as output:
                        downloaded = offset
                        last_report = time.perf_counter()
                        while block := response.read(4*1024*1024):
                            output.write(block)
                            downloaded += len(block)
                            if time.perf_counter()-last_report > 15:
                                print(json.dumps({'file': spec['name'], 'downloaded_mb': round(downloaded/1e6),
                                                  'total_mb': round(spec['bytes']/1e6)}), flush=True)
                                last_report = time.perf_counter()
                if part.stat().st_size != spec['bytes'] or digest(part) != spec['sha256']:
                    raise RuntimeError(f'Official size/hash mismatch: {spec["name"]}')
                part.replace(target)
                break
            except Exception as exc:
                print(json.dumps({'file': spec['name'], 'attempt': attempt+1, 'error': str(exc)}), flush=True)
                if attempt == 2:
                    raise
        print(json.dumps({'verified': spec['name'], 'elapsed_seconds': round(time.perf_counter()-start, 2)}), flush=True)
    return {**spec, 'local_path': str(target.relative_to(ROOT)).replace('\\', '/'), 'verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['1.7b', '4b', 'both'], default='4b')
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    wanted = [item for item in ARTIFACTS if item['kind'] == 'runner' or args.model in ('1.7b', 'both')]
    if args.model in ('4b', 'both'):
        wanted.append(MODEL_4B)
    with ThreadPoolExecutor(max_workers=3) as pool:
        checked = list(pool.map(download, wanted))
    runtime = BASE / 'runtime'
    runtime.mkdir(exist_ok=True)
    for item in checked:
        if item['kind'] == 'runner':
            with zipfile.ZipFile(ROOT / item['local_path']) as archive:
                for entry in archive.infolist():
                    dest = (runtime / entry.filename).resolve()
                    if not dest.is_relative_to(runtime.resolve()):
                        raise ValueError('Unsafe archive path')
                    if entry.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif dest.exists():
                        with archive.open(entry) as handle:
                            expected = hashlib.file_digest(handle, 'sha256').hexdigest()
                        if digest(dest) != expected:
                            raise RuntimeError(f'Existing runtime differs from verified archive: {entry.filename}')
                    else:
                        archive.extract(entry, runtime)
    manifest = BASE / ('artifacts.json' if args.model == '1.7b' else 'installation-'+args.model+'.json')
    manifest.write_text(json.dumps({'model_revisions': {'1.7b': REVISION, '4b': MODEL_4B_REVISION},
                                    'artifacts': checked}, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'runtime': str(runtime), 'executables': [str(p) for p in runtime.rglob('llama-server.exe')]}), flush=True)


if __name__ == '__main__':
    main()
