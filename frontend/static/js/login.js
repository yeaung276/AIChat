const form = document.getElementById("login-form");
const submitBtn = document.getElementById("submit-btn");
const errorMessage = document.getElementById("error-message");
const loading = document.getElementById("loading");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const username = document.getElementById("username").value.trim();
  const password = document.getElementById("password").value;

  if (!username || !password) {
    showError("Please fill in all fields");
    return;
  }

  submitBtn.disabled = true;
  loading.hidden = false;
  errorMessage.hidden = true;

  try {
    const response = await fetch("/api/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Login failed");
    }

    // Login successful, redirect to main app
    window.location.href = "/";
  } catch (error) {
    showError(error.message);
    submitBtn.disabled = false;
  } finally {
    loading.hidden = true;
  }
});

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}
