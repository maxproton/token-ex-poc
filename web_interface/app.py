"""Flask chat UI that routes every prompt through tea_v1's Guardrail before it reaches the LLM."""
import secrets
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

# "Vector 1 - Padding" has spaces so it can't be imported as a package; put it on sys.path instead.
VECTOR1_DIR = Path(__file__).resolve().parent.parent / "Vector 1 - Padding"
sys.path.insert(0, str(VECTOR1_DIR))

from tea_v1.guardrail import Guardrail, THRESHOLD  # noqa: E402

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

guardrail = Guardrail()


@app.route("/")
def index():
    return render_template("index.html", threshold=THRESHOLD, model=guardrail.cfg.model)


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    session_id = session.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id

    allowed, score, _parts, _decision, _classifier_info = guardrail.check(message, session_id)
    if not allowed:
        return jsonify({"blocked": True, "score": round(score, 2),
                         "error": "This prompt failed our guard rail checks."})

    result = guardrail.ask(message)
    if result["status"] != "ok":
        return jsonify({"blocked": False, "error": f"Model error: {result.get('error')}"}), 502

    return jsonify({"blocked": False, "score": round(score, 2), "reply": result["text"]})


if __name__ == "__main__":
    app.run(debug=True)
