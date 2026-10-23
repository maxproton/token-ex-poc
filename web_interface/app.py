"""Flask chat UI that routes every prompt through tea_v1's, tea_v2's, tea_v3's, tea_v4's and
tea_v5's Guardrails before it reaches the LLM, and through tea_v2's streaming output guard (with
tea_v5's predicted reasoning ceiling plugged in when enabled) while it is being generated."""
import secrets
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, session

# All five vector folders have spaces in their names so they can't be imported as packages; put
# them on sys.path instead. tea_v1/tea_v2/tea_v3/tea_v4/tea_v5 each define a module-level
# `guardrail`, so they're imported under distinct names to avoid shadowing each other.
VECTOR1_DIR = Path(__file__).resolve().parent.parent / "Vector 1 - Padding"
VECTOR2_DIR = Path(__file__).resolve().parent.parent / "Vector 2 - Recursive Prompting"
VECTOR3_DIR = Path(__file__).resolve().parent.parent / "Vector 3 - Context Stuffing"
VECTOR4_DIR = Path(__file__).resolve().parent.parent / "Vector 4 - Multi-turn Accumulation"
VECTOR5_DIR = Path(__file__).resolve().parent.parent / "Vector 5 - Complexity Inflation"
sys.path.insert(0, str(VECTOR1_DIR))
sys.path.insert(0, str(VECTOR2_DIR))
sys.path.insert(0, str(VECTOR3_DIR))
sys.path.insert(0, str(VECTOR4_DIR))
sys.path.insert(0, str(VECTOR5_DIR))

from tea_v1.guardrail import Guardrail as GuardrailV1, THRESHOLD as THRESHOLD_V1  # noqa: E402
from tea_v2.guardrail import Guardrail as GuardrailV2, THRESHOLD as THRESHOLD_V2  # noqa: E402
from tea_v3.guardrail import Guardrail as GuardrailV3, THRESHOLD as THRESHOLD_V3  # noqa: E402
from tea_v3.guardrail import SYSTEM_PROMPT as SYSTEM_PROMPT_V3  # noqa: E402
from tea_v3.config import RetrieverConfig  # noqa: E402
from tea_v4.guardrail import Guardrail as GuardrailV4, THRESHOLD as THRESHOLD_V4  # noqa: E402
from tea_v5.guardrail import Guardrail as GuardrailV5, THRESHOLD as THRESHOLD_V5  # noqa: E402

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

guardrail_v1 = GuardrailV1()
guardrail_v2 = GuardrailV2()
# corpus_dir is relative to tea_v3's own folder by default (for its standalone CLI); give it an
# absolute path here since the Flask app isn't run from "Vector 3 - Context Stuffing/".
guardrail_v3 = GuardrailV3(retriever_cfg=RetrieverConfig(corpus_dir=str(VECTOR3_DIR / "tea_v3" / "corpus")))
guardrail_v4 = GuardrailV4()
guardrail_v5 = GuardrailV5()

# Per-session conversation history (list of {"role", "content"} turns), resent to the model with
# every request when enable_history is on -- without this, V4's session-accumulation detection has
# nothing real to measure, since the model would never actually see a growing conversation. Stored
# server-side rather than in the signed Flask session cookie since it grows unbounded per turn.
conversation_histories = {}


@app.route("/")
def index():
    threshold = (f"{THRESHOLD_V1} (V1) / {THRESHOLD_V2} (V2) / {THRESHOLD_V3} (V3) / "
                 f"{THRESHOLD_V4} (V4) / {THRESHOLD_V5} (V5)")
    return render_template("index.html", threshold=threshold, model=guardrail_v1.cfg.model)


@app.route("/clear", methods=["POST"])
def clear():
    """Wipe this session's state across every vector that carries any: V4's accumulation
    history (prompt/completion trend + cumulative budget), V1's accepted-prompt baseline, and the
    resent conversation history. Without this there's no way to get back to a true clean slate
    short of restarting the server or opening a new browser session."""
    session_id = session.get("session_id")
    if session_id:
        conversation_histories.pop(session_id, None)
        guardrail_v1.reset_session(session_id)
        guardrail_v4.reset_session(session_id)
    return jsonify({"ok": True})


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400
    # Lets the UI route a message straight to the model without RAG retrieval/scoring at all, for
    # comparing guarded vs unguarded behavior against V3's context-stuffing attacks specifically.
    enable_v3 = bool(body.get("enable_v3", True))
    # Same idea for V4's session-accumulation tracking: off means this turn is never scored
    # against the session's history and never folded into it either, so toggling it back on
    # later resumes from whatever history had already accumulated while it was off.
    enable_v4 = bool(body.get("enable_v4", True))
    # Off means this turn is neither sent with prior history nor appended to it, same pattern as
    # V3/V4 above; toggling back on resumes from whatever history had already accumulated.
    enable_history = bool(body.get("enable_history", True))
    # Off means no classifier call and no live reasoning-ceiling abort on this turn's generation --
    # the model is asked exactly as it would be with V5 never having existed.
    enable_v5 = bool(body.get("enable_v5", True))

    session_id = session.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        session["session_id"] = session_id

    history = conversation_histories.setdefault(session_id, []) if enable_history else []

    allowed_v1, score_v1, _parts, _decision, _classifier_info = guardrail_v1.check(message, session_id)
    if not allowed_v1:
        return jsonify({"blocked": True, "score": round(score_v1, 2),
                         "error": "This prompt failed our guard rail checks."})

    allowed_v2, score_v2, _parts, _decision = guardrail_v2.check(message, session_id)
    if not allowed_v2:
        return jsonify({"blocked": True, "score": round(score_v2, 2),
                         "error": "This prompt failed our guard rail checks."})

    if enable_v4:
        # V4 scores this turn's prompt size against this session's own recent turns, not against
        # the message content: a message can look completely benign on its own and still be the
        # Nth turn of a conversation that's been quietly growing every turn.
        allowed_v4, score_v4, _parts, _decision = guardrail_v4.check(message, session_id)
        if not allowed_v4:
            return jsonify({"blocked": True, "score": round(score_v4, 2),
                             "error": "This prompt failed our guard rail checks."})
    else:
        score_v4 = 0.0

    if enable_v3:
        # V3 retrieves local knowledge-base context for the message and scores the retrieved
        # chunks, not the message itself: a message can look completely benign and still pull in a
        # poisoned, duplicated or oversized document from the corpus.
        allowed_v3, score_v3, _parts, _decision, chunks = guardrail_v3.check(message)
        if not allowed_v3:
            return jsonify({"blocked": True, "score": round(score_v3, 2),
                             "error": "This prompt failed our guard rail checks."})
        context = guardrail_v3.assemble_context(message, chunks)
    else:
        score_v3 = 0.0
        context = message

    predicted_ceiling_tokens = None
    if enable_v5:
        # V5 scores the message's own structure (bundled asks, unbounded scope, reasoning bait)
        # and, if that passes, predicts a reasoning-token ceiling via the classifier -- the real
        # check only fires once generation streams back and can be compared against that
        # prediction below.
        allowed_v5, score_v5_input, _parts, _decision, predicted_ceiling_tokens = guardrail_v5.check(message)
        if not allowed_v5:
            return jsonify({"blocked": True, "score": round(score_v5_input, 2),
                             "error": "This prompt failed our guard rail checks."})
    else:
        score_v5_input = 0.0

    # Generation goes through V2's streaming output guard over the RAG-assembled context (message
    # plus accepted retrieved chunks), so a reply that starts looping or runs far past what the
    # question needed is still cut off mid-stream, and the model answers using retrieved context.
    # Prior conversation turns (if enabled) are resent ahead of this turn's context so the model
    # actually sees a growing conversation, not just the current message in isolation. When V5 is
    # on, its predicted reasoning-token ceiling rides along on this same streaming call rather than
    # opening a second connection for the same reply.
    result = guardrail_v2.ask(context, system=SYSTEM_PROMPT_V3, history=history,
                               reasoning_ceiling_tokens=predicted_ceiling_tokens)
    if result["status"] == "error":
        return jsonify({"blocked": False, "error": f"Model error: {result.get('error')}"}), 502

    if enable_v4:
        # Only known once the model has actually responded; feeds the next turn's
        # completion-growth and cumulative-budget checks for this session.
        guardrail_v4.record_completion(session_id, result.get("completion_tokens"))

    score_v5 = score_v5_input
    if enable_v5:
        # The real V5 signal: how far actual reasoning ran past the predicted ceiling, only known
        # now that generation has finished (or been cut short).
        resp_score_v5, _resp_parts = guardrail_v5.score_response(result)
        score_v5 = max(score_v5_input, resp_score_v5)

    if enable_history:
        # Store the plain message/reply, not the RAG-stuffed context, so resent history reads like
        # an ordinary conversation rather than re-including retrieved documents on every turn.
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": result["text"]})

    response = {"blocked": False,
                "score": round(max(score_v1, score_v2, score_v3, score_v4, score_v5), 2),
                "reply": result["text"]}
    if result["status"] == "truncated":
        response["truncated"] = True
        response["abort_reason"] = result["abort_reason"]
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
