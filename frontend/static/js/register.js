const form = document.getElementById("register-form");
const submitBtn = document.getElementById("submit-btn");
const errorMessage = document.getElementById("error-message");
const loading = document.getElementById("loading");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const username = document.getElementById("username").value.trim();
  const name = document.getElementById("name").value.trim();
  const bio = document.getElementById("bio").value.trim();
  const password = document.getElementById("password").value;

  if (!username || !name || !password) {
    showError("Please fill in all required fields");
    return;
  }

  if (username.length < 3) {
    showError("Username must be at least 3 characters");
    return;
  }

  if (password.length < 6) {
    showError("Password must be at least 6 characters");
    return;
  }

  submitBtn.disabled = true;
  loading.hidden = false;
  errorMessage.hidden = true;

  try {
    const response = await fetch("/api/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, name, bio, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Registration failed");
    }

    // Registration successful, redirect to main app
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
