const MESSAGE_TYPE_SDP_ANSWER = "SDP_ANSWER";
const MESSAGE_TYPE_SDP_OFFER = "SDP_OFFER";
const MESSAGE_TYPE_AVATAR_INITIALIZE = "AVATAR_INITIALIZE";
const MESSAGE_TYPE_AVATAR_SPEAK = "AVATAR_SPEAK";
const MESSAGE_TYPE_AVATAR_INTERRUPT = "AVATAR_INTERRUPT";
const MESSAGE_TYPE_TRANSCRIPT = "SPEECH_TRANSCRIPT"

const ICE_CONFIG = {
  iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
};

class Communicator {
  constructor() {
    this.onAvatarSpeak = () => null;
    this.onAvatarInterrupt = () => null;
    this.onAvatarInitialize = () => null;
    this.onTranscript = () => null;
  }

  async setup() {
    return new Promise((resolve, reject) => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      this.ws = new WebSocket(`${protocol}//${window.location.host}/api/ws`);

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
    chatId,
    onAvatarInitialize = () => null,
    onAvatarSpeak = () => null,
    onAvatarInterrupt = () => null,
    onTranscript = () => null
  ) {
    if (!chatId) {
      throw new Error("Chat ID is required");
    }

    this.onAvatarInitialize = onAvatarInitialize;
    this.onAvatarSpeak = onAvatarSpeak;
    this.onAvatarInterrupt = onAvatarInterrupt;
    this.onTranscript = onTranscript;

    this.rtc = new RTCPeerConnection(ICE_CONFIG);

    return new Promise(async (resolve, reject) => {
      // Get user media
      this.localStreams = await navigator.mediaDevices.getUserMedia({
        video: true,
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
          chat_id: chatId,
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
    this.onTranscript = null;
    this.onAvatarInterrupt = null;
    this.onAvatarSpeak = null;
    this.onAvatarInitialize = null;
  }

  async handleWSMessage(event) {
    const message = JSON.parse(event.data);

    if (message.type === MESSAGE_TYPE_SDP_ANSWER) {
      await this.rtc.setRemoteDescription(
        new RTCSessionDescription({ type: "answer", sdp: message.sdp })
      );
    }

    if (message.type == MESSAGE_TYPE_AVATAR_INITIALIZE) {
      this.onAvatarInitialize && this.onAvatarInitialize(message.data);
    }
    
    if (message.type === MESSAGE_TYPE_AVATAR_SPEAK) {
      this.onAvatarSpeak && this.onAvatarSpeak(message.data);
    }

    if (message.type === MESSAGE_TYPE_AVATAR_INTERRUPT) {
      this.onAvatarInterrupt && this.onAvatarInterrupt(message.data);
    }

    if (message.type == MESSAGE_TYPE_TRANSCRIPT) {
      this.onTranscript && this.onTranscript(message.data);
    }
  }
}

window.Communicator = Communicator;
