| **Category** | **Details** |
|-------------|-------------|
| **Inference Engine** | https://github.com/k2-fsa/sherpa-onnx |
| **Model Repository** | sherpa-onnx/asr-models |
| **Model Download Path** | `curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2` |
| **Original Model** | https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed |
| **Implementation Reference** | https://github.com/ruzhila/voiceapi/blob/main/README.md |
| **Model size** | 487M |
| **example** | tests/gpu/zipformer.py | 

## Advantages
- Easy to use with ONNX optimization.
- Production ready framework
- A lot of quantized models with both online and offline transcribing.
- Support multiple language.
- Support automatic resampling to targeted sample rate.