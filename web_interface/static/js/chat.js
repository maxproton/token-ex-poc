const log = document.getElementById("chat-log");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(text, cssClass) {
  const el = document.createElement("div");
  el.className = `msg ${cssClass}`;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

async function send() {
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  if (input.parentElement.MaterialTextfield) {
    input.parentElement.MaterialTextfield.change();
  }

  appendMessage(message, "user");
  const pending = appendMessage("I'm thinking a second...", "bot");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    if (data.error) {
      pending.textContent = data.error;
      pending.className = "msg error";
    } else {
      pending.textContent = data.reply;
    }
  } catch (err) {
    pending.textContent = `Request failed: ${err}`;
    pending.className = "msg error";
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
