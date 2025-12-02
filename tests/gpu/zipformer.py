# Reference: https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/speech-recognition-from-microphone.py
import asyncio

from aichat.stt.zipformer import ZipformerSTT

MODEL_DIR = "/Users/yeaung/Workspace/AIChat/aichat/stt/zipformer/model"
TEST_AUDIO = "/Users/yeaung/Workspace/AIChat/tests/audios/voice-sample.mp3"



async def main():
    ZipformerSTT.configure(sr=16000, model_dir='')
    stt = ZipformerSTT()

    print("Started! Please speak")

    sample_rate = 48000
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = await asyncio.to_thread(s.read, samples_per_read)  # a non-blocking read
            samples = samples.reshape(-1)
            stt.accept(samples)
            


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")