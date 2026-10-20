"""CodeAlpha AI Internship - Task 2: FAQ Chatbot (web app)

    pip install -r requirements.txt
    python app.py            -> open http://127.0.0.1:5000
"""
import json
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from faq_engine import FAQBot

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
bot = FAQBot(HERE / "knowledge_base.json", use_embeddings=False if os.getenv("USE_EMBEDDINGS") == "0" else "auto")


def _log(name: str, record: dict):
    record["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_DIR / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/")
def index():
    return render_template("index.html", meta=bot.meta)


@app.get("/api/meta")
def meta():
    return jsonify({"bot": bot.meta, "topics": bot.topics(), "faq_count": len(bot.faqs),
                    "semantic": bot.embedder is not None})


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))[:500]
    result = bot.ask(message, context_id=data.get("context_id"))
    if result.get("unanswered"):
        _log("unanswered.jsonl", {"message": message, "score": result["confidence"]})
    return jsonify(result)


@app.get("/api/faq/<faq_id>")
def faq(faq_id):
    result = bot.get(faq_id)
    return (jsonify(result), 200) if result else (jsonify({"error": "unknown FAQ"}), 404)


@app.post("/api/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    _log("feedback.jsonl", {"message": str(data.get("message", ""))[:500], "faq_id": data.get("faq_id"),
                            "helpful": bool(data.get("helpful"))})
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, port=int(os.getenv("PORT", 5000)))
