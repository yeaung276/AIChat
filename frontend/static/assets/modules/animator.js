// Reference: https://github.com/met4citizen/HeadTTS/blob/main/modules/worker-tts.mjs

import { TalkingHead } from "talkinghead";
import { StyleTextToSpeech2Model, AutoTokenizer, Tensor } from "transformer";
import { Language } from "../language/en-us.js";

const TTS_MODEL = "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped";
const TTS_DICTIONARY = "/static/assets/dictionaries/en-us.txt";
const TTS_STYLE_DIM = 256;
const TTS_FRAME_RATE = 40;
const TTS_DELTA_START = -10;
const TTS_DELTA_END = 10;
const TTS_SAMPLING_RATE = 24000;

class HeadAnimator {
  constructor(node) {
    this.node = node;
    this.head = new TalkingHead(node, {
      ttsEndpoint: "N/A",
      cameraView: "upper",
      mixerGainSpeech: 3,
      cameraRotateEnable: false,
    });
  }

  async setup() {
    console.log(`loading styleTTS model...`);
    this.ttsModel = await StyleTextToSpeech2Model.from_pretrained(TTS_MODEL, {
      dtype: "fp32",
      device: "webgpu",
    });

    console.log(`loading tokenizer...`);
    this.ttsTokenizer = await AutoTokenizer.from_pretrained(TTS_MODEL);

    console.log(`loading languages...`);
    this.language = new Language();
    await this.language.loadDictionary(TTS_DICTIONARY);
  }

  async start(avatar, voice) {
    console.log(`loading avatar and voice...`);

    const loadVoice = async () => {
      const response = await fetch(voice.url);
      if (!response.ok) {
        console.error("fail to fetch voice.");
        return;
      }
      this.ttsVoice = new Float32Array(await response.arrayBuffer());
    };

    const loadAvatar = async () => {
      this.node.style.display = "block";
      await this.head.showAvatar(avatar);
    };

    await Promise.all([loadVoice(), loadAvatar()]);
  }

  async stop() {
    console.log("stopping the avatar...");
    this.node.style.display = "none";
    await this.head.stop();
  }

  async speak(text, speed = 1) {
    // phoneme and tokenization
    const { phonemes, metadata, silences } = this.language.generate(text);
    const { input_ids } = this.ttsTokenizer(phonemes.join(""), {
      truncation: true,
    });

    // generating audio and lip sync timestamps
    const offset =
      Math.min(Math.max(input_ids.size - 1, 0), 509) * TTS_STYLE_DIM;

    const { waveform, durations } = await this.ttsModel({
      input_ids,
      style: new Tensor(
        "float32",
        this.ttsVoice.slice(offset, offset + TTS_STYLE_DIM),
        [1, TTS_STYLE_DIM]
      ),
      speed: new Tensor("float32", [speed], [1]),
    });
    this.updateLipSyncTimestamps(
      metadata,
      Array.from(durations.data),
      silences
    );
    const samples = silences.length
      ? this.insertSilences(waveform.data, TTS_SAMPLING_RATE, silences)
      : waveform.data;

    // encoding
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    metadata.audio = await audioCtx.decodeAudioData(
      this.encodePCM(samples, TTS_SAMPLING_RATE, true)
    );
    metadata.audioEncoding = "pcm";
    this.head.speakAudio(metadata, {});
  }

  updateLipSyncTimestamps(o, ds, silences) {
    // Calculate starting times in milliseconds
    const scaler = 1000 / TTS_FRAME_RATE; // From frames to milliseconds
    const times = [];
    let t = 0;
    let len = ds.length;
    for (let i = 0; i < len; i++) {
      times.push(Math.round(t));
      t += scaler * ds[i];
    }
    times.push(Math.round(t)); // Last entry

    // Shift times based on silent periods and
    // convert phoneme indexes to original starting times
    const shifts = silences.map((x) => [x[0] + 1, x[1]]);
    silences.forEach((x) => (x[0] = times[x[0]] - 20));
    shifts.forEach((x) => {
      for (let i = x[0]; i < times.length; i++) {
        times[i] += x[1];
      }
    });

    // Calculate word times and durations (+1 because of $)
    len = o.words.length;
    for (let i = 0; i < len; i++) {
      const start = times[o.wtimes[i] + 1] + TTS_DELTA_START;
      const end = times[o.wdurations[i] + 1] + TTS_DELTA_END;
      const duration = end - start;
      o.wtimes[i] = start;
      o.wdurations[i] = duration;
    }

    // Calculate visemes times and durations (+1 because of $)
    len = o.visemes.length;
    for (let i = 0; i < len; i++) {
      const start = times[o.vtimes[i] + 1] + TTS_DELTA_START;
      const end = times[o.vdurations[i] + 1] + TTS_DELTA_END;
      const duration = end - start;
      o.vtimes[i] = start;
      o.vdurations[i] = duration;
    }
  }

  insertSilences(samples, samplerate, silences) {
    // Convert times and durations to number of samples
    let nNewSamples = 0; // Total new samples
    silences.forEach((x) => {
      x[0] = Math.floor((x[0] / 1000) * samplerate);
      x[1] = Math.floor((x[1] / 1000) * samplerate);
      nNewSamples += x[1];
    });

    // New Float32Array
    const result = new Float32Array(samples.length + nNewSamples);

    // Copy existing samples
    let readPos = 0;
    let writePos = 0;
    silences.forEach((x) => {
      const start = Math.min(x[0], samples.length);
      const len = start - readPos;
      if (len > 0) {
        result.set(samples.subarray(readPos, start), writePos);
        readPos += len;
        writePos += len;
      }
      writePos += x[1]; // Add silence
    });
    if (readPos < samples.length) {
      result.set(samples.subarray(readPos), writePos);
    }

    return result;
  }

  encodePCM(samples, sampleRate, header = true) {
    const len = samples.length;
    let offset = header ? 44 : 0;
    const buffer = new ArrayBuffer(offset + len * 2);
    const view = new DataView(buffer);

    // Write WAV header
    if (header) {
      function writeString(view, off, string) {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(off + i, string.charCodeAt(i));
        }
      }

      writeString(view, 0, "RIFF");
      view.setUint32(4, 32 + samples.length * 2, true);
      writeString(view, 8, "WAVE");
      writeString(view, 12, "fmt ");
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(view, 36, "data");
      view.setUint32(40, samples.length * 2, true);
    }

    // Write samples as PCM 16bit LE
    for (let i = 0; i < len; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return buffer;
  }
}

window.HeadAnimator = HeadAnimator;
