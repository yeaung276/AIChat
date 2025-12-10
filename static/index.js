// core interaction
const dialBtn = document.getElementById("dial");
const dropBtn = document.getElementById("drop");
const avatar = document.getElementById("avatar");

// core modules
const animator = new window.HeadAnimator(avatar);
const communicator = new window.Communicator();

// event listeners
window.onload = async () => {
  await communicator.setup();
  await animator.setup();
  removeLoading();
  console.log("initialization complete.");
};

dialBtn.addEventListener("click", async () => {
  console.log("dialing...");
  showLoading();
  await communicator.start(
    (message) => animator.speak(message),
    () => console.log("speech interrupted")
  );
  await animator.start();
  toggleControl("dialing")
  removeLoading();
  console.log("ready.");
});

dropBtn.addEventListener("click", async () => {
  await communicator.stop();
  await animator.stop();
  toggleControl("free")
});

// util functions
function showLoading() {
  if (document.getElementById("aura-loading")) return;

  const aura = document.createElement("div");
  aura.id = "aura-loading";

  aura.innerHTML = `
    <div class="aura-ring"></div>
  `;

  document.body.appendChild(aura);
}

function removeLoading() {
  const aura = document.getElementById("aura-loading");

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
