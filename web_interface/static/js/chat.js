const log = document.getElementById("chat-log");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const v3Toggle = document.getElementById("v3-toggle");
const v4Toggle = document.getElementById("v4-toggle");
const historyToggle = document.getElementById("history-toggle");
const v5Toggle = document.getElementById("v5-toggle");
const clearHistoryBtn = document.getElementById("clear-history-btn");

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
  const pending = appendMessage("...", "bot");

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        enable_v3: v3Toggle.checked,
        enable_v4: v4Toggle.checked,
        enable_history: historyToggle.checked,
        enable_v5: v5Toggle.checked,
      }),
    });
    const data = await res.json();

    if (data.error) {
      pending.textContent = data.error;
      pending.className = "msg error";
    } else if (data.truncated) {
      pending.textContent = `${data.reply}\n\n[response cut short mid-stream: ${data.abort_reason}]`;
    } else {
      pending.textContent = data.reply;
    }
  } catch (err) {
    pending.textContent = `Request failed: ${err}`;
    pending.className = "msg error";
  }
}

async function clearHistory() {
  clearHistoryBtn.disabled = true;
  try {
    await fetch("/clear", { method: "POST" });
    log.innerHTML = "";
  } catch (err) {
    appendMessage(`Failed to clear history: ${err}`, "error");
  } finally {
    clearHistoryBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
clearHistoryBtn.addEventListener("click", clearHistory);
