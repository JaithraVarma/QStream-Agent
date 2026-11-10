"""Download the pinned public Qwen files into this project."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


HERE = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
MODEL_DIR = Path(os.environ.get("QSTREAM_MODEL_DIR", HERE / ".model")).expanduser().resolve()
FILES = ("config.json", "generation_config.json", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-local", action="store_true")
    args = parser.parse_args()
    if not args.verify_local:
        snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                          allow_patterns=list(FILES), token=False)
    files = {name: (MODEL_DIR / name).stat().st_size for name in FILES}
    digest = hashlib.sha256()
    with (MODEL_DIR / "model.safetensors").open("rb") as weights:
        for chunk in iter(lambda: weights.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("The weights checksum does not match the pinned snapshot")
    receipt = dict(model=MODEL_ID, revision=REVISION, file_bytes=files,
                   weights_sha256=digest.hexdigest())
    (HERE / "model-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
