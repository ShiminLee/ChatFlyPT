"""Fetch the llama.cpp organisation's Q8 conversion of official Qwen Instruct 2507."""
import importlib.util
import json
from pathlib import Path

source = Path(__file__).with_name('local-model-install.py')
spec = importlib.util.spec_from_file_location('local_model_install', source)
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)
REVISION = 'e6f794d44f9395d0184a966c27b5ae99ea356fcb'
ARTIFACT = {
    'name': 'qwen3-4b-instruct-2507-q8_0.gguf', 'kind': 'model', 'bytes': 4280403520,
    'sha256': 'ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1',
    'url': f'https://huggingface.co/ggml-org/Qwen3-4B-Instruct-2507-Q8_0-GGUF/resolve/{REVISION}/qwen3-4b-instruct-2507-q8_0.gguf?download=true',
}

if __name__ == '__main__':
    checked = install.download(ARTIFACT)
    (install.BASE/'artifacts-4b.json').write_text(json.dumps({
        'model_revision': REVISION, 'source_model': 'Qwen/Qwen3-4B-Instruct-2507',
        'gguf_publisher': 'ggml-org (llama.cpp organisation), not the Qwen account',
        'artifact': checked}, indent=2), encoding='utf-8')
