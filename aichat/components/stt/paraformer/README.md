| **Category** | **Details** |
|-------------|-------------|
| **Inference Engine** | https://github.com/k2-fsa/sherpa-onnx |
| **Model Repository** | sherpa-onnx/asr-models |
| **Model Download Path** | `curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2` |
| **Original Model** | https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en |
| **Implementation Reference** | https://github.com/ruzhila/voiceapi/blob/main/README.md |
| **Model size** | ~220M |
| **example** | tests/gpu/paraformer.py | 

## Advantages
- Easy to use with ONNX optimization.
- Production ready framework
- A lot of quantized models with both online and offline transcribing.
- Support multiple language (bilingual Chinese-English support).
- Support automatic resampling to targeted sample rate.
- Faster inference speed compared to Zipformer with comparable accuracy.
- Lower memory footprint due to non-autoregressive architecture.
- Better for real-time streaming applications with lower latency.