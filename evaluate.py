"""Measure the chatbot on held-out questions and compare it with the original version.

    python evaluate.py            # summary table
    python evaluate.py --misses   # also list every question the new engine got wrong

The "original" engine is the first version of this project re-run on the same,
larger knowledge base: NLTK preprocessing + one TF-IDF cosine + threshold 0.08.
"""
import argparse
import json
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_engine import FAQBot
from nlp import STOPWORDS, lemmatize, tokenize

HERE = Path(__file__).resolve().parent


class OriginalBot:
    """The first version of the engine, kept for comparison."""

    def __init__(self, faqs, threshold=0.08):
        self.faqs, self.threshold = faqs, threshold
        self.vec = TfidfVectorizer(ngram_range=(1, 2))
        self.m = self.vec.fit_transform([self.pre(f"{f['question']} {f['question']} {f['answer']} {f['keywords']}")
                                         for f in faqs])

    @staticmethod
    def pre(text):
        text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        return " ".join(lemmatize(t) for t in tokenize(text) if t not in STOPWORDS)

    def predict(self, q):
        s = cosine_similarity(self.vec.transform([self.pre(q)]), self.m)[0]
        b = int(s.argmax())
        return (self.faqs[b]["id"], "confident") if s[b] >= self.threshold else (None, "declined")


def run(predict, cases):
    """predict(q) -> (faq_id or None, mode) where mode is 'confident', 'hedged' or 'declined'."""
    in_scope = [c for c in cases if c["expected"]]
    oos = [c for c in cases if not c["expected"]]
    top1 = confident_right = confident_wrong = 0
    misses = []
    for c in in_scope:
        got, mode = predict(c["q"])
        if got == c["expected"]:
            top1 += 1
            confident_right += mode == "confident"
        else:
            misses.append((c["q"], c["expected"], got, mode))
            confident_wrong += mode == "confident"
    oos_confident = 0
    for c in oos:
        got, mode = predict(c["q"])
        oos_confident += mode == "confident"
        if mode != "declined":
            misses.append((c["q"], None, got, mode))
    n, m = len(in_scope), max(len(oos), 1)
    return {
        "Right FAQ picked (top-1)": top1 / n,
        "Right FAQ, answered confidently": confident_right / n,
        "Wrong FAQ, answered confidently  (lower is better)": confident_wrong / n,
        "Off-topic answered confidently   (lower is better)": oos_confident / m,
    }, misses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--misses", action="store_true")
    ap.add_argument("--embeddings", action="store_true", help="also use sentence-transformers")
    args = ap.parse_args()

    cases = json.loads((HERE / "tests" / "eval_questions.json").read_text())["cases"]
    t0 = time.time()
    bot = FAQBot(use_embeddings=args.embeddings)
    build = time.time() - t0
    old = OriginalBot(bot.faqs)

    def new_predict(q):
        r = bot.ask(q)
        mode = {"answer": "confident", "multi": "confident", "clarify": "hedged"}.get(r["type"], "declined")
        return r["faq_id"], mode

    old_res, _ = run(old.predict, cases)
    t0 = time.time()
    new_res, misses = run(new_predict, cases)
    per_q = (time.time() - t0) / len(cases) * 1000

    n_in = sum(1 for c in cases if c["expected"])
    print(f"\n{len(cases)} held-out questions ({n_in} in scope, {len(cases) - n_in} out of scope), "
          f"{len(bot.faqs)} FAQs\n")
    print(f"{'metric':52}{'original':>10}{'new':>10}")
    for k in old_res:
        print(f"{k:52}{old_res[k]:>10.1%}{new_res[k]:>10.1%}")
    print(f"\nindex build {build:.2f}s · {per_q:.1f} ms per question\n")
    if args.misses:
        for q, exp, got, mode in misses:
            print(f"  x {q!r}: expected {exp}, got {got} ({mode})")


if __name__ == "__main__":
    main()
