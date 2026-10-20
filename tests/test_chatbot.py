"""Tests for the FAQ chatbot.  Run from the project folder:  python -m pytest -v"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from faq_engine import FAQBot  # noqa: E402
from nlp import Preprocessor, normalise  # noqa: E402


@pytest.fixture(scope="module")
def bot():
    return FAQBot(ROOT / "knowledge_base.json", use_embeddings=False)


@pytest.fixture(scope="module")
def client():
    import app as web
    web.app.config["TESTING"] = True
    return web.app.test_client()


# ---------------------------------------------------------------- preprocessing
def test_normalise_expands_contractions_and_slang():
    assert normalise("Where's my parcel??") == "where is my parcel"
    assert normalise("I can't login, pls help") == "i can not login please help"
    assert "how" in normalise("sign up kaise kare")


def test_spelling_correction_uses_shop_vocabulary(bot):
    p = bot.pre.process("how to cancle my ordr")
    assert p["corrections"].get("cancle") == "cancel"
    assert "cancel" in p["tokens"]


def test_real_words_are_not_corrected(bot):
    assert bot.pre.process("recommend a good movie")["corrections"] == {}


def test_synonyms_map_to_one_term():
    pre = Preprocessor({"refund": ["money back"], "upi": ["gpay"]})
    assert pre.process("I want my money back on gpay", correct_spelling=False)["tokens"] == ["refund", "upi"]


def test_negation_is_kept(bot):
    assert "not" in bot.pre.process("I have not received it")["tokens"]


# ---------------------------------------------------------------- answering
@pytest.mark.parametrize("question, expected", [
    ("wheres my parcle?", "track-order"),
    ("how to cancle order", "cancel-order"),
    ("forgot pasword", "reset-password"),
    ("is delivery free", "shipping-charges"),
    ("can i use phonepe", "payment-methods"),
    ("you sent me the wrong product", "damaged-item"),
    ("let me speak to a real person", "contact-support"),
])
def test_matches_paraphrased_questions(bot, question, expected):
    r = bot.ask(question)
    assert r["type"] in ("answer", "clarify")
    assert r["faq_id"] == expected


@pytest.mark.parametrize("question", ["what is the capital of france", "write me a poem about cats",
                                      "how do i cook biryani", "asdkjh qwe"])
def test_off_topic_questions_are_not_answered_confidently(bot, question):
    assert bot.ask(question)["type"] in ("fallback", "clarify")


def test_small_talk(bot):
    assert bot.ask("hi")["intent"] == "greeting"
    assert bot.ask("thanks a lot")["intent"] == "thanks"
    assert bot.ask("are you a bot?")["intent"] == "identity"


def test_greeting_plus_question_answers_the_question(bot):
    r = bot.ask("hello! how do i cancel my order")
    assert r["faq_id"] == "cancel-order"
    assert r["answer"].startswith("Hi there!")


def test_two_questions_in_one_message(bot):
    r = bot.ask("How do I return this and when do I get my refund?")
    assert r["type"] == "multi"
    assert len(r["parts"]) == 2


def test_follow_up_uses_previous_answer(bot):
    r = bot.ask("what about cards?", context_id="refund-time")
    assert r["faq_id"] == "refund-time"
    assert r["debug"]["context_used"]


def test_card_number_triggers_warning(bot):
    r = bot.ask("my card 4111 1111 1111 1111 payment failed")
    assert r.get("warning")
    assert "4111" not in r["debug"]["normalised"]


def test_clicked_faq_returns_its_answer(bot):
    r = bot.get("cod")
    assert r["faq_id"] == "cod" and r["confidence"] == 1.0
    assert bot.get("does-not-exist") is None


def test_knowledge_base_is_consistent(bot):
    ids = [f["id"] for f in bot.faqs]
    assert len(ids) == len(set(ids)), "duplicate FAQ ids"
    for f in bot.faqs:
        assert f["answer"].strip() and f["question"].strip()
        for r in f["related"]:
            assert r in bot.by_id, f"{f['id']} links to unknown FAQ {r}"


def test_accuracy_does_not_regress(bot):
    """Held-out set: at least 85% top-1 and no confidently-answered off-topic questions."""
    import json
    cases = json.loads((ROOT / "tests" / "eval_questions.json").read_text())["cases"]
    in_scope = [c for c in cases if c["expected"]]
    hits = sum(bot.ask(c["q"])["faq_id"] == c["expected"] for c in in_scope)
    assert hits / len(in_scope) >= 0.85
    assert all(bot.ask(c["q"])["type"] != "answer" for c in cases if not c["expected"])


# ---------------------------------------------------------------- web API
def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Sunny" in r.data


def test_api_meta(client):
    data = client.get("/api/meta").get_json()
    assert data["faq_count"] == 50
    assert [t["category"] for t in data["topics"]][0] == "Orders"


def test_api_chat_and_feedback(client):
    r = client.post("/api/chat", json={"message": "track my order"}).get_json()
    assert r["faq_id"] == "track-order" and r["confidence_label"] == "high"
    assert client.post("/api/feedback", json={"faq_id": "track-order", "helpful": True}).get_json()["ok"]


def test_api_faq_lookup(client):
    assert client.get("/api/faq/cod").status_code == 200
    assert client.get("/api/faq/nope").status_code == 404


def test_api_handles_empty_and_bad_input(client):
    assert client.post("/api/chat", json={}).status_code == 200
    assert client.post("/api/chat", data="not json").status_code == 200


@pytest.mark.parametrize("message, previous, expected", [
    ("what about express?", "delivery-time", "express-delivery"),   # moves to a related topic
    ("what about cod", "payment-methods", "cod"),                    # drills into a listed option
    ("and if it's broken?", "exchange", "damaged-item"),            # switches topic
    ("what about gift cards?", "refund-time", "gift-card"),         # clearly a new topic
    ("what about my password", "track-order", "reset-password"),
])
def test_follow_ups_can_change_topic(bot, message, previous, expected):
    assert bot.ask(message, context_id=previous)["faq_id"] == expected
