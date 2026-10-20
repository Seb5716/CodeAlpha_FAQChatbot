"""Text preprocessing pipeline for the FAQ chatbot (NLTK based).

    raw text
      -> normalise      (unicode, lowercase, contractions, chat slang, a little Hinglish)
      -> tokenise       (NLTK word_tokenize, regex fallback)
      -> spell-correct  (only against the shop's own vocabulary, so real words are never "fixed")
      -> synonyms       (multi-word phrases and single words mapped to one canonical term)
      -> stopwords      (NLTK list, but negations like "not" are kept)
      -> lemmatise      (WordNet: noun form, then verb form)

Everything degrades gracefully: if NLTK data can't be downloaded the pipeline
falls back to a regex tokenizer, a built-in stopword list and no lemmatisation.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

# ---------------------------------------------------------------- NLTK setup
try:
    import nltk

    def _ensure(resource: str, package: str) -> bool:
        for name in (resource, resource + ".zip"):   # corpora are often left zipped
            try:
                nltk.data.find(name)
                return True
            except LookupError:
                pass
        try:
            return bool(nltk.download(package, quiet=True))
        except Exception:
            return False

    _HAS_PUNKT = _ensure("tokenizers/punkt_tab", "punkt_tab") or _ensure("tokenizers/punkt", "punkt")
    _HAS_STOP = _ensure("corpora/stopwords", "stopwords")
    _HAS_WORDNET = _ensure("corpora/wordnet", "wordnet")
except ImportError:  # NLTK not installed at all
    nltk = None
    _HAS_PUNKT = _HAS_STOP = _HAS_WORDNET = False

if _HAS_STOP:
    from nltk.corpus import stopwords as _sw
    _BASE_STOPWORDS = set(_sw.words("english"))
else:
    _BASE_STOPWORDS = set(
        "i me my myself we our ours you your yours he him his she her it its they them their what which who whom "
        "this that these those am is are was were be been being have has had having do does did doing a an the and "
        "but if or because as until while of at by for with about against between into through during before after "
        "above below to from up down in out on off over under again further then once here there when where why how "
        "all any both each few more most other some such only own same so than too very s t can will just should now".split())

if _HAS_WORDNET:
    from nltk.corpus import wordnet as _wn
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
else:
    _wn = None
    _lemmatizer = None

# Negations change meaning ("not received"), so they stay in.
_KEEP = {"not", "no", "nor", "never", "same"}
# Chat filler that carries no meaning for matching.
_EXTRA_STOP = {"please", "pls", "plz", "kindly", "hi", "hello", "hey", "thanks", "thank", "sir", "madam", "mam",
               "shopeasy", "sunny", "want", "wanna", "need", "would", "like", "get", "got", "know", "tell", "u",
               "also", "anymore", "times", "yesterday", "today", "still", "something", "thing", "one"}
STOPWORDS = (_BASE_STOPWORDS | _EXTRA_STOP) - _KEEP

# ---------------------------------------------------------------- normalisation tables
_CONTRACTIONS = [
    (r"\bwon'?t\b", "will not"), (r"\bcan'?t\b", "can not"), (r"\bcannot\b", "can not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|could|should|would)n'?t\b", r"\1 not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"\bim\b", "i am"),
    (r"\b(where|what|how|who|when|that|it|there|here)'s\b", r"\1 is"),
    (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"), (r"'s\b", ""),
]
_SLANG = {
    "wheres": "where is", "whats": "what is", "hows": "how is", "whens": "when is", "thats": "that is",
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "thx": "thanks", "ty": "thanks",
    "wanna": "want to", "gonna": "going to", "gimme": "give me", "asap": "urgently", "abt": "about",
    "msg": "message", "no.": "number", "num": "number", "nos": "numbers", "acc": "account", "acct": "account",
    "pwd": "password", "pw": "password", "amt": "amount", "addr": "address", "info": "information",
    "bday": "birthday", "pkg": "package", "qty": "quantity", "cust": "customer",
    # common Hinglish words seen in Indian support chats
    "kaise": "how", "kese": "how", "kab": "when", "kahan": "where", "kaha": "where", "kya": "what",
    "mera": "my", "meri": "my", "mujhe": "i", "paisa": "money", "paise": "money", "wapas": "back",
    "kare": "do", "karna": "do", "karu": "do", "nahi": "not", "nhi": "not", "hai": "", "kar": "do",
}


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def normalise(text: str) -> str:
    """Lowercase, unify quotes, expand contractions, replace slang, strip punctuation."""
    text = _strip_accents(text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("₹", " rs ").replace("&", " and ")
    for pat, rep in _CONTRACTIONS:
        text = re.sub(pat, rep, text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = [_SLANG.get(w, w) for w in text.split()]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def tokenize(text: str) -> list[str]:
    if _HAS_PUNKT:
        try:
            return nltk.word_tokenize(text)
        except LookupError:
            pass
    return re.findall(r"[a-z0-9]+", text)


def lemmatize(token: str) -> str:
    if _lemmatizer is None or not token.isalpha():
        return token
    noun = _lemmatizer.lemmatize(token, "n")
    return noun if noun != token else _lemmatizer.lemmatize(token, "v")


def is_english_word(token: str) -> bool:
    return bool(_wn and _wn.synsets(token))


class Preprocessor:
    """Holds the domain synonym table and vocabulary used for spell correction."""

    def __init__(self, synonyms: dict[str, list[str]] | None = None):
        self.phrase_map: list[tuple[re.Pattern, str]] = []   # multi-word -> canonical
        self.word_map: dict[str, str] = {}                     # single word -> canonical
        for canonical, alts in (synonyms or {}).items():
            for alt in alts:
                alt_n = normalise(alt)
                if " " in alt_n:
                    self.phrase_map.append((re.compile(rf"\b{re.escape(alt_n)}\b"), canonical))
                elif alt_n:
                    self.word_map[alt_n] = canonical
        # longest phrases first so "cash on delivery" wins over "delivery"
        self.phrase_map.sort(key=lambda p: -len(p[0].pattern))
        self.vocab: set[str] = set(self.word_map) | set(self.word_map.values())

    def learn_vocab(self, texts):
        """Add the knowledge base's own words to the spell-correction vocabulary."""
        for t in texts:
            self.vocab.update(w for w in normalise(t).split() if w.isalpha())

    def correct(self, token: str) -> str:
        """Fix a typo only if the word is unknown both to the shop vocabulary and to English."""
        if (len(token) < 4 or not token.isalpha() or token in self.vocab
                or token in STOPWORDS or is_english_word(token)):
            return token
        match = difflib.get_close_matches(token, self.vocab, n=1, cutoff=0.78)
        return match[0] if match else token

    def process(self, text: str, correct_spelling: bool = True) -> dict:
        """Run the full pipeline. Returns tokens plus a record of corrections (for the UI)."""
        norm = normalise(text)
        corrections = {}
        if correct_spelling:
            fixed = []
            for w in norm.split():
                c = self.correct(w)
                if c != w:
                    corrections[w] = c
                fixed.append(c)
            norm = " ".join(fixed)
        for pat, canonical in self.phrase_map:
            norm = pat.sub(canonical, norm)
        tokens = []
        for tok in tokenize(norm):
            if tok.isdigit():               # a 6-digit number is an Indian pincode; other numbers carry no topic
                if len(tok) == 6:
                    tokens.append("pincode")
                continue
            tok = self.word_map.get(tok, tok)
            if tok in STOPWORDS or not tok.strip():
                continue
            lem = lemmatize(tok)
            tokens.append(self.word_map.get(lem, lem))
        return {"normalised": norm, "tokens": tokens, "text": " ".join(tokens), "corrections": corrections}
