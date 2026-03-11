# AIChat

A real-time AI avatar chat application. Users talk to an AI avatar via webcam and microphone — the avatar listens, understands emotions from your face, generates an empathetic response, and speaks back using a synthesized voice.

## Features

- **Real-time voice conversation** via WebRTC — no push-to-talk, continuous streaming
- **Emotion-aware responses** — DeepFace reads facial expressions and informs the LLM context
- **Fine-tuned LLM** — Qwen2.5 or TinyLlama with LoRA adapters trained on empathetic dialogue
- **Text-to-speech** — Kokoro TTS synthesizes the avatar's voice
- **3D avatar** — animated GLB avatar speaks in sync with the audio
- **Multi-user** — JWT authentication, per-user chat history, feedback collection
- **Pluggable components** — swap STT/LLM/TTS/emotion models via `config.yaml` without code changes

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A GPU is recommended for running the LLM locally (CPU works but is slow)

---

## Installation

```bash
# Clone and install dependencies
uv sync
```

For production (adds vllm for faster inference):
```bash
uv sync --group prod
```

---

## Download Models

LoRA weights are bundled in `assets/`. Run this once to unpack them and download the STT model:

```bash
uv run python scripts/download.py
```

This sets up:
- `models/qwen2.5-lora` — fine-tuned Qwen2.5 LoRA adapter
- `models/tiny-llama-lora` — fine-tuned TinyLlama LoRA adapter
- `models/zipformer` — streaming ASR model (downloaded from GitHub releases)

---

## Running the App

```bash
uv run uvicorn aichat.app:app --reload --port 8080
```

Then open `http://localhost:8080` in your browser.

**First time:**
1. Go to `/register` and create an account
2. Log in at `/login`
3. Start a chat session — allow microphone and camera access when prompted
4. The avatar will respond in real time as you speak

### Environment variables (`.env`)

| Variable | Description |
|---|---|
| `CONFIG_FILE` | Path to config file (default: `config.yaml`) |
| `NGROK_API_KEY` | Optional — exposes the app publicly via ngrok |
| `NGROK_APP_PORT` | Port to expose via ngrok |
---

## Configuration

All AI components are configured in `config.yaml`. You can swap implementations without touching code.

```yaml
speech:          # STT model
emotion:         # Facial emotion analyzer
llm:             # Language model
voice:           # TTS model
avatars:         # Available 3D faces and voice presets
```

Each component entry needs a `name`, a `path` (dotted import: `module.path:ClassName`), and optional `config`. The `name` is what you reference when starting a chat session.

Use `dummy` implementations during development — they return empty/mock outputs instantly:

```yaml
llm:
  - name: dummy
    path: "aichat.components.llm.dummy:DummyLLM"
    config:
```

---

## Project Structure

```
aichat/
  app.py              # FastAPI app, startup lifecycle
  components/         # Pluggable AI components
    stt/              # Speech-to-text (zipformer, paraformer, dummy)
    llm/              # Language models (transformer, dummy)
    tts/              # Text-to-speech (kokoro, orpheus, dummy)
    video/            # Emotion analysis (deepface, dummy)
  pipeline/
    factory.py        # Loads and wires components from config
    processor.py      # Per-session audio/video/LLM/TTS pipeline
    manager.py        # WebRTC connection manager
    context.py        # Conversation context / prompt builder
  routes/
    chat.py           # Chat, WebSocket (SDP exchange), feedback API
    user.py           # Auth endpoints (register, login)
  db_models/          # SQLModel database models
  schemas/            # Pydantic request/response schemas
  security/           # JWT auth

frontend/             # Pure HTML/JS/CSS, no build step
  index.html          # Main chat UI
  login.html
  register.html
  feedback.html
  static/             # Avatars, voices, JS, CSS

training/             # LoRA fine-tuning scripts (Colab)
experiments/          # Offline evaluation scripts
scripts/              # Utility scripts (model download)
data/                 # Training and evaluation data
models/               # Downloaded/unpacked model weights
assets/               # Bundled LoRA zip files
reports/              # Evaluation output plots
```

---

## Testing

```bash
uv run pytest --cov=aichat --cov-report=html tests/

# View coverage report in browser
open htmlcov/index.html
```

---

## Adding a New Component

1. Create a class implementing the relevant protocol in `aichat/components/*/base.py`
2. Register it in `config.yaml`:

```yaml
llm:
  - name: my-model
    path: "aichat.components.llm.my_module:MyLLM"
    config:
      model: "org/model-name"
```

All components must implement a `configure(**kwargs)` classmethod called at startup.

---

## Training

Fine-tuning runs on Google Colab using [Unsloth](https://github.com/unslothai/unsloth).

**Dataset:** [Empathetic Dialogues (Facebook AI)](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai/data)

```bash
# Prepare training data
uv run python -m training.data
```

Then run `training/train.py` in Colab. Trained adapters save to Google Drive and are zipped into `assets/`.

---

## Evaluation

Offline scripts in `experiments/` for measuring model quality. Results saved to `reports/`.

| Script | What it measures |
|---|---|
| `experiments/evaluate_llm.py` | Perplexity, BERTScore, emotion F1, latency across LLM models |
| `experiments/evaluate_stt.py` | WER, RTF, endpoint latency across STT models |
| `experiments/evaluate_tts.py` | TTS latency vs input length, RTF |
| `experiments/evaluate_pipeline.py` | End-to-end component latency profiling |

```bash
uv run python -m experiments.evaluate_llm
uv run python -m experiments.evaluate_stt
uv run python -m experiments.evaluate_tts
uv run python -m experiments.evaluate_pipeline
```
