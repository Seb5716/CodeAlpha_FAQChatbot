"""FAQ matching engine.

For every user message the bot:
  1. handles small talk (greetings, thanks, "are you a bot?") and warns if the
     user is about to share an OTP / card number;
  2. splits messages that hold two questions ("how do I return this and when
     do I get my refund?") and answers both;
  3. preprocesses the text with NLTK (see nlp.py);
  4. scores every FAQ with four signals and blends them:
        word TF-IDF cosine  vs. the FAQ question and its paraphrases (max)
        char n-gram cosine  (robust to leftover typos and word forms)
        answer TF-IDF cosine vs. the answer + keywords
        keyword coverage    share of the user's IDF-weighted words the FAQ knows
     plus, optionally, sentence-embedding similarity (sentence-transformers);
  5. turns the blended score into a decision: answer / ask "did you mean?" /
     admit it doesn't know - instead of always returning the top match;
  6. uses the previous answer as context for short follow-ups ("what about COD?").
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp import Preprocessor, normalise

HERE = Path(__file__).resolve().parent

# Decision thresholds on the blended score (0-1). Set by looking at the score
# distribution of tests/eval_questions.json - see evaluate.py.
ANSWER_AT = 0.45          # confident: answer directly ...
ANSWER_AT_COVERED = 0.33  # ... or a bit lower when most of the user's words are known to that FAQ
COVERED = 0.50
CLARIFY_AT = 0.27         # plausible: answer, but offer alternatives ("did you mean")
MARGIN = 0.02             # if #1 and #2 are this close and unrelated, offer both

WEIGHTS = {"question": 0.40, "char": 0.20, "answer": 0.15, "coverage": 0.25}

SMALL_TALK = [
    ("greeting", r"^(hi+|hello+|hey+|hiya|yo|namaste|hola|good (morning|afternoon|evening))\b[\s!,.]*"),
    ("thanks", r"\b(thanks|thank you|thank u|thx|ty|dhanyavad|shukriya|appreciate it)\b"),
    ("bye", r"^(bye|goodbye|good bye|see you|see ya|that is all|thats all|nothing else|no thanks)\b"),
    ("identity", r"\b(who|what) are you\b|\bare you (a |an )?(bot|human|robot|real|ai)\b|\byour name\b"),
    ("capabilities", r"^(help|menu|options|start)$|\bwhat can you (do|help)"),
    ("ack", r"^(ok|okay|k|cool|great|nice|got it|fine|alright|perfect|awesome|done)[\s!.]*$"),
    ("frustration", r"\b(useless|stupid|dumb|not helpful|wrong answer|you are bad|doesn ?t help|didn ?t help)\b"),
]

SMALL_TALK_REPLIES = {
    "greeting": "Hello! What can I help you with today? You can ask about an order, a delivery, a return or your account.",
    "thanks": "You're welcome! Anything else I can help with?",
    "bye": "Thanks for chatting with ShopEasy. Have a great day!",
    "identity": "I'm **Sunny**, ShopEasy's automated help assistant. I match your question against our help articles. For anything I can't solve, I'll point you to a human on our support team.",
    "capabilities": "I can answer questions about **orders, delivery, returns & refunds, payments, offers and your account**. Type a question in your own words, or pick a topic on the left.",
    "ack": "Great! Let me know if there's anything else.",
    "frustration": "Sorry I didn't get that right. Could you rephrase it with a bit more detail? You can also reach a person at **1800-123-456** (Mon–Sat, 9am–6pm) or **support@shopeasy.example**.",
}

PII_PATTERNS = [
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "card number"),
    (re.compile(r"\b(otp|cvv|pin|password)\b\D{0,12}\d{3,8}\b", re.I), "OTP / PIN"),
]
PII_WARNING = ("**Please don't share card numbers, OTPs, PINs or passwords in chat.** "
               "ShopEasy will never ask for them.")

_SPLIT_RE = re.compile(
    r"\?+\s*|\s+(?:and also|and|also)\s+(?=(?:how|what|when|where|why|can|could|do|does|is|are|will|should|which)\b)",
    re.I)

# "what about COD?", "and if it's broken?" -> read together with the previous answer
_FOLLOW_UP_RE = re.compile(r"^(and|what about|how about|what if|and if|same for|also|then)\b")


class FAQBot:
    def __init__(self, kb_path: str | Path = HERE / "knowledge_base.json", use_embeddings: bool | str = "auto"):
        kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
        if isinstance(kb, list):              # old format: a plain list of FAQs
            kb = {"bot": {}, "synonyms": {}, "faqs": kb}
        self.meta = kb.get("bot", {})
        self.faqs = kb["faqs"]
        for i, f in enumerate(self.faqs):
            f.setdefault("id", f"faq-{i}")
            f.setdefault("category", "General")
            f.setdefault("variants", [])
            f.setdefault("keywords", "")
            f.setdefault("related", [])
        self.by_id = {f["id"]: f for f in self.faqs}
        self.categories = list(dict.fromkeys(f["category"] for f in self.faqs))

        self.pre = Preprocessor(kb.get("synonyms", {}))
        self.pre.learn_vocab(t for f in self.faqs
                             for t in [f["question"], *f["variants"], f["keywords"]])
        self._build_index()
        self.embedder = self._load_embedder() if use_embeddings else None
        if use_embeddings is True and self.embedder is None:
            raise RuntimeError("sentence-transformers is not installed")

    # ------------------------------------------------------------ indexing
    def _build_index(self):
        rows, owner = [], []
        for idx, f in enumerate(self.faqs):
            for q in [f["question"], *f["variants"]]:
                rows.append(self.pre.process(q, correct_spelling=False)["text"])
                owner.append(idx)
        self.row_owner = np.array(owner)
        answer_docs = [self.pre.process(f["question"] + " " + f["keywords"] + " " + f["answer"],
                                        correct_spelling=False)["text"] for f in self.faqs]

        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.word_vec.fit(rows + answer_docs)
        self.q_matrix = self.word_vec.transform(rows)
        self.a_matrix = self.word_vec.transform(answer_docs)

        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_matrix = self.char_vec.fit_transform(rows)

        # token sets per FAQ for the coverage signal (question + variants + keywords)
        self.faq_tokens = [set() for _ in self.faqs]
        for r, i in zip(rows, owner):
            self.faq_tokens[i].update(r.split())
        for i, f in enumerate(self.faqs):
            self.faq_tokens[i].update(self.pre.process(f["keywords"], correct_spelling=False)["tokens"])
        vocab = self.word_vec.vocabulary_
        idf = self.word_vec.idf_
        self.idf = {t: idf[j] for t, j in vocab.items() if " " not in t}
        self.unknown_idf = float(idf.max()) + 1.0   # words the shop has never seen weigh the most
        self.kb_words = set(self.idf)

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            return None
        texts, owner = [], []
        for idx, f in enumerate(self.faqs):
            for q in [f["question"], *f["variants"]]:
                texts.append(q)
                owner.append(idx)
        self.emb_owner = np.array(owner)
        self.emb_matrix = model.encode(texts, normalize_embeddings=True)
        return model

    # ------------------------------------------------------------ scoring
    def _max_per_faq(self, row_scores: np.ndarray, owner: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.faqs))
        np.maximum.at(out, owner, row_scores)
        return out

    def score(self, text: str, processed: dict | None = None) -> tuple[np.ndarray, dict, dict]:
        """Return (blended scores per FAQ, per-signal scores, preprocessing info)."""
        p = processed or self.pre.process(text)
        if not p["tokens"]:
            zeros = np.zeros(len(self.faqs))
            return zeros, {k: zeros for k in WEIGHTS}, p
        q_text = p["text"]
        s_question = self._max_per_faq(cosine_similarity(self.word_vec.transform([q_text]), self.q_matrix)[0],
                                       self.row_owner)
        s_char = self._max_per_faq(cosine_similarity(self.char_vec.transform([q_text]), self.char_matrix)[0],
                                   self.row_owner)
        s_answer = cosine_similarity(self.word_vec.transform([q_text]), self.a_matrix)[0]

        weights = [self.idf.get(t, self.unknown_idf) for t in p["tokens"]]
        total = sum(weights)
        s_cov = np.array([sum(w for t, w in zip(p["tokens"], weights) if t in toks) / total
                          for toks in self.faq_tokens])

        signals = {"question": s_question, "char": s_char, "answer": s_answer, "coverage": s_cov}
        blended = sum(WEIGHTS[k] * v for k, v in signals.items())
        if self.embedder is not None:
            emb = self.embedder.encode([text], normalize_embeddings=True)
            s_sem = self._max_per_faq(self.emb_matrix @ emb[0], self.emb_owner)
            signals["semantic"] = s_sem
            blended = 0.5 * blended + 0.5 * np.clip((s_sem - 0.2) / 0.6, 0, 1)
        return blended, signals, p

    # ------------------------------------------------------------ public API
    def ask(self, message: str, context_id: str | None = None) -> dict:
        """Answer one chat message. `context_id` = id of the FAQ answered last turn."""
        message = (message or "").strip()
        if not message:
            return self._reply("smalltalk", self.meta.get("greeting", "Hi! How can I help?"), intent="empty")

        warning = None
        if any(p.search(message) for p, _ in PII_PATTERNS):
            warning = PII_WARNING
            message = PII_PATTERNS[0][0].sub(" ", message)

        # small talk - only when nothing else is left after removing it
        norm = normalise(message)
        greeted = False
        m = re.match(SMALL_TALK[0][1], norm)
        if m and norm[m.end():].strip():
            greeted, message = True, re.sub(r"^\W*\w+[\s!,.]*", "", message, count=1) or message
            norm = normalise(message)
        intent = self._small_talk(norm)
        if intent and (intent != "thanks" or len(self.pre.process(message)["tokens"]) <= 1):
            return self._reply("smalltalk", SMALL_TALK_REPLIES[intent], intent=intent, warning=warning)

        # multi-question messages
        parts = [s.strip() for s in _SPLIT_RE.split(message) if s and len(s.strip()) > 3]
        if len(parts) > 1:
            answers = [self._answer_one(pt, None) for pt in parts]
            good = [a for a in answers if a["type"] == "answer"]
            if len(good) > 1 and len({a["faq_id"] for a in good}) == len(good):
                body = "\n\n".join(f"**{a['matched_question']}**\n{a['answer']}" for a in good)
                res = self._reply("multi", body, faq_id=good[-1]["faq_id"],
                                  confidence=min(a["confidence"] for a in good),
                                  parts=[{"faq_id": a["faq_id"], "question": a["matched_question"],
                                          "confidence": a["confidence"]} for a in good],
                                  related=[r for a in good for r in a["related"]
                                           if r["id"] not in {g["faq_id"] for g in good}][:3],
                                  debug=good[0]["debug"])
                return self._finish(res, greeted, warning)

        return self._finish(self._answer_one(message, context_id), greeted, warning)

    def _small_talk(self, norm: str) -> str | None:
        for name, pat in SMALL_TALK:
            if re.search(pat, norm):
                if name == "greeting" and norm[re.match(pat, norm).end():].strip():
                    continue
                return name
        return None

    def _answer_one(self, message: str, context_id: str | None) -> dict:
        scores, signals, p = self.score(message)
        order = np.argsort(-scores)
        best, second = int(order[0]), int(order[1])
        top, gap = float(scores[best]), float(scores[best] - scores[second])
        context_used = False

        # short follow-up? try again with the previous topic as context
        known = [t for t in p["tokens"] if t in self.kb_words]
        follow_up = bool(_FOLLOW_UP_RE.match(p["normalised"]))
        if (context_id in self.by_id and known and len(p["tokens"]) <= 4
                and (follow_up or top < ANSWER_AT_COVERED)):
            prev = self.by_id[context_id]
            s_ctx, sig_ctx, _ = self.score(message + " " + prev["question"] + " " + prev["keywords"])
            # the new message still dominates, so "what about express?" after a delivery
            # question moves to the express FAQ instead of repeating the old answer
            combined = 0.6 * scores + 0.4 * s_ctx
            if follow_up:   # "what about cards?" most likely stays on the same topic or a related one
                for fid in [context_id, *prev["related"]]:
                    if fid in self.by_id:
                        combined[self.faqs.index(self.by_id[fid])] += 0.12 if fid == context_id else 0.05
                # every word of the follow-up already appears in the previous answer -> stay there,
                # unless the direct match is a related FAQ (drilling down: "payment methods" -> "what about COD?")
                prev_answer = set(self.pre.process(prev["answer"], correct_spelling=False)["tokens"])
                if set(p["tokens"]) <= prev_answer and self.faqs[best]["id"] not in prev["related"]:
                    combined[self.faqs.index(prev)] += 0.10
            o2 = np.argsort(-combined)
            if combined[o2[0]] >= ANSWER_AT_COVERED and (follow_up or combined[o2[0]] > top + 0.05):
                context_used = int(o2[0]) != best or top < ANSWER_AT_COVERED
                scores, order = combined, o2
                signals = {k: 0.6 * signals[k] + 0.4 * sig_ctx[k] for k in signals}
                best, second = int(o2[0]), int(o2[1])
                top, gap = float(combined[best]), float(combined[best] - combined[second])

        debug = {
            "normalised": p["normalised"], "tokens": p["tokens"], "corrections": p["corrections"],
            "signals": {k: round(float(v[best]), 3) for k, v in signals.items()},
            "context_used": context_used,
        }
        alts = [{"id": self.faqs[i]["id"], "question": self.faqs[i]["question"], "score": round(float(scores[i]), 3)}
                for i in order[1:4] if scores[i] >= CLARIFY_AT * 0.8]

        if top < CLARIFY_AT:
            suggestions = [{"id": self.faqs[i]["id"], "question": self.faqs[i]["question"]}
                           for i in order[:3] if scores[i] >= CLARIFY_AT * 0.6]
            return self._reply("fallback", self.meta.get("fallback", "Sorry, I don't know that one yet."),
                               confidence=top, alternatives=suggestions, debug=debug, unanswered=True)

        faq = self.faqs[best]
        covered = signals["coverage"][best] >= COVERED
        confident = top >= ANSWER_AT or (top >= ANSWER_AT_COVERED and covered)
        if not confident or (gap < MARGIN and alts and self.faqs[second]["id"] not in faq["related"]
                               and scores[second] >= CLARIFY_AT):
            kind = "clarify"
        else:
            kind = "answer"
        return self._reply(kind, faq["answer"], faq_id=faq["id"], matched_question=faq["question"],
                           category=faq["category"], confidence=top,
                           alternatives=alts if kind == "clarify" else [],
                           related=[{"id": r, "question": self.by_id[r]["question"]}
                                    for r in faq["related"] if r in self.by_id],
                           debug=debug)

    def _finish(self, res: dict, greeted: bool, warning: str | None) -> dict:
        if greeted and res["type"] in ("answer", "clarify", "multi"):
            res["answer"] = "Hi there! " + res["answer"]
        if warning:
            res["warning"] = warning
        return res

    @staticmethod
    def _reply(kind: str, answer: str, **extra) -> dict:
        conf = float(extra.pop("confidence", 1.0 if kind == "smalltalk" else 0.0))
        label = {"answer": "high", "multi": "high", "smalltalk": "high", "clarify": "medium"}.get(kind, "low")
        base = {"type": kind, "answer": answer, "confidence": round(conf, 3), "confidence_label": label,
                "faq_id": None, "matched_question": None, "category": None,
                "alternatives": [], "related": [], "debug": {}}
        base.update(extra)
        return base

    def topics(self) -> list[dict]:
        """FAQs grouped by category, in knowledge-base order."""
        return [{"category": c, "items": [{"id": f["id"], "question": f["question"]}
                                          for f in self.faqs if f["category"] == c]}
                for c in self.categories]

    def get(self, faq_id: str) -> dict | None:
        """Answer for a FAQ chosen by clicking (no matching needed)."""
        f = self.by_id.get(faq_id)
        if not f:
            return None
        return self._reply("answer", f["answer"], faq_id=f["id"], matched_question=f["question"],
                           category=f["category"], confidence=1.0,
                           related=[{"id": r, "question": self.by_id[r]["question"]}
                                    for r in f["related"] if r in self.by_id])


# ---------------------------------------------------------------- terminal chat
if __name__ == "__main__":
    bot = FAQBot()
    print(f"{bot.meta.get('name', 'Bot')} · ShopEasy help (type 'quit' to exit)\n")
    last = None
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        r = bot.ask(q, context_id=last)
        last = r["faq_id"] or last
        print(f"Bot [{r['type']}, {r['confidence']:.2f}]: {r['answer']}")
        if r.get("alternatives"):
            print("     Did you mean: " + " | ".join(a["question"] for a in r["alternatives"]))
        print()
