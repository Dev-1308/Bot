let currentLang = "hi";
let currentRole = "Grower";

const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const roleSelector = document.getElementById("roleSelector");
const langToggle = document.getElementById("langToggle");

roleSelector.addEventListener("change", () => {
  currentRole = roleSelector.value;
});

function toggleLanguage() {
  currentLang = currentLang === "en" ? "hi" : "en";
  langToggle.innerText = currentLang === "en" ? "Hindi" : "English";
}

function appendMessage(text, sender) {
  const msgDiv = document.createElement("div");
  msgDiv.className = `message ${sender}`;
  msgDiv.innerText = text;
  chatBox.appendChild(msgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  appendMessage(message, "user");
  userInput.value = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      role: currentRole,
      lang: currentLang,
    }),
  });

  const data = await response.json();
  appendMessage(data.reply, "bot");
}
