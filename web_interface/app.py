"""Flask chat UI that routes every prompt through tea_v1's and tea_v2's Guardrails before it
reaches the LLM, and through tea_v2's streaming output guard while it is being generated."""
import secrets
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

# Both vector folders have spaces in their names so they can't be imported as packages; put them
# on sys.path instead. tea_v1 and tea_v2 both define a module-level `guardrail`, so they're
# imported under distinct names to avoid shadowing each other.
VECTOR1_DIR = Path(__file__).resolve().parent.parent / "Vector 1 - Padding"
VECTOR2_DIR = Path(__file__).resolve().parent.parent / "Vector 2 - Recursive Prompting"
sys.path.insert(0, str(VECTOR1_DIR))
sys.path.insert(0, str(VECTOR2_DIR))

from tea_v1.guardrail import Guardrail as GuardrailV1, THRESHOLD as THRESHOLD_V1  # noqa: E402
from tea_v2.guardrail import Guardrail as GuardrailV2, THRESHOLD as THRESHOLD_V2  # noqa: E402

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

guardrail_v1 = GuardrailV1()
guardrail_v2 = GuardrailV2()


@app.route("/")
def index():
    return render_template("index.html", threshold=f"{THRESHOLD_V1} (V1) / {THRESHOLD_V2} (V2)",
                            model=guardrail_v1.cfg.model)


@app.route("/chat", methods=["POST"])
def chat():
    message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    session_id = session.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id

    allowed_v1, score_v1, _parts, _decision, _classifier_info = guardrail_v1.check(message, session_id)
    if not allowed_v1:
        return jsonify({"blocked": True, "score": round(score_v1, 2),
                         "error": "This prompt failed our guard rail checks."})

    allowed_v2, score_v2, _parts, _decision = guardrail_v2.check(message, session_id)
    if not allowed_v2:
        return jsonify({"blocked": True, "score": round(score_v2, 2),
                         "error": "This prompt failed our guard rail checks."})

    # Generation itself goes through V2's streaming output guard, so a reply that starts
    # looping or runs far past what the question needed is still cut off mid-stream.
    result = guardrail_v2.ask(message)
    if result["status"] == "error":
        return jsonify({"blocked": False, "error": f"Model error: {result.get('error')}"}), 502

    response = {"blocked": False, "score": round(max(score_v1, score_v2), 2), "reply": result["text"]}
    if result["status"] == "truncated":
        response["truncated"] = True
        response["abort_reason"] = result["abort_reason"]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
