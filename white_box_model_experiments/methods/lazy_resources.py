"""Lazily-constructed shared resources.

Importing an attack module used to eagerly download and instantiate
``bert-base-uncased``, a ``fill-mask`` pipeline, ``all-MiniLM-L6-v2`` and four
NLTK corpora - roughly 500 MB of downloads - even for a PAIR or SRA run that
never touches any of them. Everything here is built on first use instead.
"""

import functools
import logging

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = "bert-base-uncased"
SENTENCE_TRANSFORMER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# NLTK corpora required by the synonym-based attack constraints.
_NLTK_REQUIREMENTS = (
    ("corpora/stopwords", "stopwords"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/wordnet", "wordnet"),
    ("tokenizers/punkt_tab", "punkt_tab"),
)


@functools.lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Download the NLTK corpora the attacks depend on, once per process."""
    import nltk

    for resource_path, package in _NLTK_REQUIREMENTS:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info("Downloading NLTK package %r", package)
            nltk.download(package, quiet=True)


@functools.lru_cache(maxsize=1)
def get_stopwords():
    """Return the English stopword set."""
    ensure_nltk_data()
    from nltk.corpus import stopwords

    return set(stopwords.words("english"))


@functools.lru_cache(maxsize=1)
def get_wordnet():
    """Return the WordNet corpus reader."""
    ensure_nltk_data()
    from nltk.corpus import wordnet

    # Force the lazy corpus loader to resolve now so failures surface here.
    wordnet.ensure_loaded()
    return wordnet


@functools.lru_cache(maxsize=1)
def get_spell_checker():
    """Return a shared ``SpellChecker`` instance."""
    from spellchecker import SpellChecker

    return SpellChecker()


@functools.lru_cache(maxsize=1)
def get_bert_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def get_bert_model():
    from transformers import AutoModelForMaskedLM

    return AutoModelForMaskedLM.from_pretrained(BERT_MODEL_NAME)


@functools.lru_cache(maxsize=1)
def get_bert_fill_mask():
    from transformers import pipeline

    return pipeline(
        "fill-mask",
        model=get_bert_model(),
        tokenizer=get_bert_tokenizer(),
        device="cpu",
    )


@functools.lru_cache(maxsize=1)
def get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SENTENCE_TRANSFORMER_NAME, device="cpu")


def word_tokenize(text: str):
    """``nltk.word_tokenize`` with the punkt corpora guaranteed to be present."""
    ensure_nltk_data()
    import nltk

    return nltk.word_tokenize(text)
