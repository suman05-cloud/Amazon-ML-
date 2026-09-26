"""Download a pinned public model without needing the GPU environment first."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "a3ded7f4d0d725c6fc251504622d65b2516a3aa1"
destination = Path("artifacts/pretrained/multilingual_minilm")
destination.mkdir(parents=True, exist_ok=True)
files = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model", "README.md", "model.safetensors"]
hashes = {}
for name in files:
    path = destination/name
    if not path.exists():
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        print(time.strftime("%H:%M:%S"), "Downloading", name, flush=True)
        temporary = path.with_suffix(path.suffix+".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as f:
            while chunk := response.read(1024*1024):
                f.write(chunk)
        temporary.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024*1024):
            digest.update(chunk)
    hashes[name] = digest.hexdigest()
(destination/"origin.json").write_text(json.dumps({"model_id": MODEL, "revision": REVISION, "license": "Apache-2.0", "sha256": hashes}, indent=2), encoding="utf-8")
print("Pinned model download complete", flush=True)
