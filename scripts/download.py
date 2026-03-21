#!/usr/bin/env python3
import shutil
import zipfile
import tarfile
import urllib.request
import asyncio
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aichat.components.llm.transformer.model import Transformer


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


def download_spacy_model():
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("spaCy model already installed, skipping.\n")
    except OSError:
        print("Downloading spaCy en_core_web_sm...")
        venv_python = Path(sys.executable).resolve()
        subprocess.run([str(venv_python), "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("Done.\n")


def download_hf_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_id = "unsloth/Qwen3-0.6B-unsloth-bnb-4bit"

    print(f"Downloading tokenizer for {model_id}...")
    AutoTokenizer.from_pretrained(model_id)
    print("Done.\n")

    print(f"Downloading model weights for {model_id}...")
    AutoModelForCausalLM.from_pretrained(model_id)
    print("Done.\n")
    
async def warmup_vllm():
    Transformer.configure(
        model="unsloth/Qwen2.5-0.5B-unsloth-bnb-4bit",
        lora_path="models/qwen2.5-lora",
        lora_name="main",
        lora_rank=8,
    )

    llm = Transformer()

    async for resp in llm.generate("Hi"):
        break


if __name__ == "__main__":
    # Qwen2.5 LoRA
    print("=== [1/6] qwen2.5-lora ===")
    copy_and_unzip("assets/qwen2.5-lora.zip", "models")

    # Tiny LLaMA LoRA
    print("=== [2/6] tiny-llama-lora ===")
    copy_and_unzip("assets/tiny-llama-lora.zip", "models")

    # Zipformer
    print("=== [3/6] zipformer ===")
    download_and_extract_tar(
        url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
        dest_dir="models/zipformer",
    )

    # spaCy
    print("=== [4/6] spaCy en_core_web_sm ===")
    download_spacy_model()

    # HuggingFace model
    print("=== [5/6] Qwen3-0.6B ===")
    download_hf_model()
    
    # Warmup vLLM
    print("=== [6/6] vLLM ===")
    download_hf_model()
    
    asyncio.run(warmup_vllm())

    print("=== All models ready ===")