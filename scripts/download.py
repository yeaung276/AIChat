#!/usr/bin/env python3
import shutil
import zipfile
import tarfile
import urllib.request
from pathlib import Path


def copy_and_unzip(src: str, dest_dir: str):
    src_path = Path(src)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    dest_zip = dest_path / src_path.name
    print(f"Copying {src_path} -> {dest_zip}")
    shutil.copy2(src_path, dest_zip)

    print(f"Unzipping into {dest_path}")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(dest_path)

    dest_zip.unlink()
    print(f"Done: {dest_path}\n")


def download_and_extract_tar(url: str, dest_dir: str):
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    dest_file = dest_path / filename

    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, dest_file, reporthook=progress_hook)
    print()

    print(f"Extracting into {dest_path}")
    with tarfile.open(dest_file, "r:bz2") as tf:
        members = tf.getmembers()
        # Strip the top-level directory from all paths
        top = members[0].name.rstrip("/")
        for member in members:
            member.path = Path(member.path).relative_to(top).as_posix()
        tf.extractall(dest_path, members=members[1:])

    dest_file.unlink()
    print(f"Done: {dest_path}\n")


def progress_hook(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        print(f"\r  {pct}%", end="", flush=True)


if __name__ == "__main__":
    # Qwen2.5 LoRA
    print("=== [1/3] qwen2.5-lora ===")
    copy_and_unzip("assets/qwen2.5-lora.zip", "models")

    # Tiny LLaMA LoRA
    print("=== [2/3] tiny-llama-lora ===")
    copy_and_unzip("assets/tiny-llama-lora.zip", "models")

    # Zipformer
    print("=== [3/3] zipformer ===")
    download_and_extract_tar(
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
        dest_dir="models/zipformer",
    )

    print("=== All models ready ===")