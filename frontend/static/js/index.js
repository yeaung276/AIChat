// DOM elements
const dialBtn = document.getElementById("dial");
const dropBtn = document.getElementById("drop");
const avatar = document.getElementById("avatar");
const transcriptBtn = document.getElementById("transcript-toggle");
const transcriptPanel = document.getElementById("transcript-panel");
const transcriptMessages = document.getElementById("transcript-messages");
const chatList = document.getElementById("chat-list");
const newChatBtn = document.getElementById("new-chat-btn");
const characterModal = document.getElementById("character-modal");
const characterForm = document.getElementById("character-form");
const modalClose = document.getElementById("modal-close");
const cancelBtn = document.getElementById("cancel-btn");
const userInfo = document.getElementById("user-info");
const logoutBtn = document.getElementById("logout-btn");

// Core modules
const animator = new window.HeadAnimator(avatar);
const communicator = new window.Communicator();

// State
let currentUser = null;
let currentChat = null;
let chats = [];

// Initialize app
window.onload = async () => {
  try {
    // Hide dial button initially until a chat is selected
    dialBtn.hidden = true;

    currentUser = await checkAuth();
    if (!currentUser) {
      window.location.href = "/login";
      return;
    }

    userInfo.textContent = `${currentUser.screen_name || currentUser.username}`;

    await loadChats();

    await communicator.setup();
    await animator.setup();

    removeLoading();
    console.log("initialization complete.");
  } catch (error) {
    console.error("Initialization error:", error);
    removeLoading();
  }
};

// Init functions
async function checkAuth() {
  try {
    const response = await fetch("/api/me");
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.error("Auth check failed:", error);
    return null;
  }
}

async function loadChats() {
  try {
    const response = await fetch("/api/chats");
    if (!response.ok) throw new Error("Failed to load chats");

    chats = await response.json();
    renderChats();
  } catch (error) {
    console.error("Error loading chats:", error);
  }
}

function renderChats() {
  chatList.innerHTML = "";

  if (chats.length === 0) {
    chatList.innerHTML = `
      <div style="padding: 20px; text-align: center; color: rgba(255, 255, 255, 0.5);">
        No chats yet. Click + to create one.
      </div>
    `;
    return;
  }

  chats.forEach((chat) => {
    const item = document.createElement("div");
    item.className = "chat-item";
    if (currentChat && currentChat.id === chat.id) {
      item.classList.add("active");
    }

    const name = document.createElement("div");
    name.className = "chat-item-name";
    name.textContent = chat.name;

    const preview = document.createElement("div");
    preview.className = "chat-item-preview";
    preview.textContent = chat.prompt || "No bio";
    preview.setAttribute("data-tooltip", chat.prompt || "No bio"); // Tooltip on hover

    item.appendChild(name);
    item.appendChild(preview);

    item.addEventListener("click", () => {
      selectChat(chat);
    });

    chatList.appendChild(item);
  });
}

// Logout
logoutBtn.addEventListener("click", async () => {
  // Clear session and redirect
  document.cookie = "session=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
  window.location.href = "/login";
});

// Transcript buttons
transcriptBtn.addEventListener("click", () => {
  transcriptPanel.classList.toggle("open");
});

function selectChat(chat) {
  currentChat = chat;
  renderChats();

  // Show dial button when chat is selected
  dialBtn.hidden = false;

  // Load transcript
  loadTranscript(chat);
}

async function loadTranscript(chat) {
  try {
    const response = await fetch(`/api/chat/${chat.id}`);
    if (!response.ok) throw new Error("Failed to load transcript");

    const chatData = await response.json();
    transcriptMessages.innerHTML = "";

    if (chatData.transcripts && chatData.transcripts.length > 0) {
      chatData.transcripts.forEach((msg) => {
        pushMessage(msg.message, msg.actor === "user" ? "right" : "left");
      });
    }
  } catch (error) {
    console.error("Error loading transcript:", error);
  }
}

// Call buttons
dialBtn.addEventListener("click", async () => {
  if (!currentChat) {
    alert("Please select a chat first");
    return;
  }

  console.log("dialing...");
  showLoading();

  try {
    await communicator.start(
      currentChat.id,
      async (data) =>
        animator.start(
          {
            url: data.avatar.url,
            body: data.avatar.gender,
            avatarMood: data.avatar.mode,
          },
          {
            url: data.voice.path,
          }
        ),
      async (data) => animator.speak(data.text),
      () => console.log("interrupted"),
      (data) =>
        pushMessage(
          data.message,
          data.actor == "user" ? "right" : "left"
        )
    );

    toggleControl("dialing");
    removeLoading();
    console.log("ready.");
  } catch (error) {
    console.error("Error dialing:", error);
    removeLoading();
    alert("Failed to start conversation: " + error.message);
  }
});

dropBtn.addEventListener("click", async () => {
  await communicator.stop();
  await animator.stop();
  toggleControl("free");

  // Reload transcript to get updated messages
  if (currentChat) {
    await loadTranscript(currentChat);
  }
});

// Character buttons
newChatBtn.addEventListener("click", () => {
  characterModal.hidden = false;
});

modalClose.addEventListener("click", () => {
  characterModal.hidden = true;
  characterForm.reset();
});

cancelBtn.addEventListener("click", () => {
  characterModal.hidden = true;
  characterForm.reset();
});

characterForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  const name = document.getElementById("char-name").value.trim();
  const voice = document.getElementById("char-voice").value;
  const face = document.getElementById("char-face").value;
  const prompt = document.getElementById("char-prompt").value.trim();

  if (!name || !prompt) {
    alert("Please fill in all required fields");
    return;
  }

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name,
        agent: { voice, face, prompt },
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Failed to create chat");
    }

    const newChat = await response.json();
    chats.push(newChat);

    characterModal.hidden = true;
    characterForm.reset();

    renderChats();
    selectChat(newChat);
  } catch (error) {
    console.error("Error creating chat:", error);
    alert("Failed to create chat: " + error.message);
  }
});

// Utility functions
function showLoading() {
  if (document.getElementById("aura-loading")) return;

  const aura = document.createElement("div");
  aura.id = "aura-loading";
  aura.innerHTML = `<div class="aura-ring"></div>`;
  document.body.appendChild(aura);
}

function removeLoading() {
  const aura = document.getElementById("aura-loading");
  if (!aura) return;

  aura.classList.add("aura-hidden");
  setTimeout(() => aura.remove(), 700);
}

function toggleControl(mode) {
  if (mode === "dialing") {
    dialBtn.hidden = true;
    dropBtn.hidden = false;
  } else if (mode === "free") {
    dialBtn.hidden = false;
    dropBtn.hidden = true;
  }
}

export function pushMessage(text, side = "left", meta = {}) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${side}`;
  bubble.textContent = text;

  if (meta && Object.keys(meta).length > 0) {
    const debug = document.createElement("div");
    debug.className = "time-debug";
    debug.textContent = Object.keys(meta).reduce(
      (p, v) => p + ` ${v}: ${meta[v]}`,
      ""
    );
    bubble.appendChild(debug);
  }

  transcriptMessages.appendChild(bubble);
  transcriptMessages.scrollTop = transcriptMessages.scrollHeight;
}
