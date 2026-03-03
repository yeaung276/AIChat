// Reference: https://github.com/met4citizen/HeadTTS/blob/main/modules/worker-tts.mjs

import { TalkingHead } from "talkinghead";

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
    console.log(`loading models...`)
  }

  async start(avatar) {
    console.log(`loading avatar and voice...`);

    this.node.style.display = "block";
    await this.head.showAvatar(avatar);
  }

  async stop() {
    console.log("stopping the avatar...");
    this.node.style.display = "none";
    await this.head.stop();
  }

  async speak(speakCtx, speed = 1) {
    const meta = {...speakCtx.meta}

    const audio = Uint8Array.from(atob(speakCtx.audio), c => c.charCodeAt(0));
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    meta.audio = await audioCtx.decodeAudioData(audio.buffer);
    meta.audioEncoding = "pcm";
    this.head.speakAudio(meta, {});
  }
}

window.HeadAnimator = HeadAnimator;
