const MESSAGE_TYPE_SDP_ANSWER = "SDP_ANSWER";
const MESSAGE_TYPE_SDP_OFFER = "SDP_OFFER"
const MESSAGE_TYPE_SPEECH_SPEAK = "SPEECH_SPEAK";
const MESSAGE_TYPE_SPEECH_INTERRUPT = "SPEECH_INTERRUPT";
const MESSAGE_TYPE_SPEECH_DEBUG = "SPEECH_DEBUG"

const ICE_CONFIG = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

class Communicator {
  constructor() {
    this.onSpeechSpeak = () => null;
    this.onSpeechInterrupt = () => null;
  }

  async setup() {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/sdp`);

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        reject();
      };

      this.ws.onclose = () => console.log("WebSocket closed");

      this.ws.onopen = async () => {
        console.log("Websocket opened.");
        resolve();
      };

      this.ws.onmessage = this.handleWSMessage.bind(this);
    });
  }

  async start(
    onSpeechSpeak = () => null, 
    onSpeechInterrupt = () => null, 
    onSpeechDebug = () => null
  ) {
    this.onSpeechSpeak = onSpeechSpeak;
    this.onSpeechInterrupt = onSpeechInterrupt;
    this.onSpeechDebug = onSpeechDebug

    this.rtc = new RTCPeerConnection(ICE_CONFIG);

    return new Promise(async (resolve, reject) => {
      // Get user media
      this.localStreams = await navigator.mediaDevices.getUserMedia({
        video: false,
        audio: true,
      });

      this.localStreams.getTracks().forEach((track) => {
        this.rtc.addTrack(track, this.localStreams);
      });

      // Create and send offer
      const offer = await this.rtc.createOffer();
      await this.rtc.setLocalDescription(offer);

      this.ws.send(
        JSON.stringify({
          type: MESSAGE_TYPE_SDP_OFFER,
          sdp: offer.sdp,
        })
      );

      this.rtc.onconnectionstatechange = (event) => {
        if (event.target.connectionState == "connected") {
          resolve();
        }
        if (event.target.connectionState == "failed") {
          reject();
        }
      };
    });
  }

  async stop() {
    if (this.rtc) {
      this.rtc.close();
      this.rtc = null;
    }

    if (this.localStreams) {
      this.localStreams.getTracks().forEach((track) => track.stop());
    }
    this.onSpeechDebug = null
    this.onSpeechInterrupt = null
    this.onSpeechSpeak = null
  }

  async handleWSMessage(event) {
    const message = JSON.parse(event.data);
    console.log("Received message:", message.type);

    if (message.type === MESSAGE_TYPE_SDP_ANSWER) {
      await this.rtc.setRemoteDescription(
        new RTCSessionDescription({ type: "answer", sdp: message.sdp })
      );
    }

    if (message.type === MESSAGE_TYPE_SPEECH_SPEAK) {
      this.onSpeechSpeak && this.onSpeechSpeak(message.data);
    }

    if (message.type === MESSAGE_TYPE_SPEECH_INTERRUPT) {
      this.onSpeechInterrupt && this.onSpeechInterrupt(message.data);
    }

    if (message.type == MESSAGE_TYPE_SPEECH_DEBUG){
      this.onSpeechDebug && this.onSpeechDebug(message.data)
    }
  }
}

window.Communicator = Communicator;