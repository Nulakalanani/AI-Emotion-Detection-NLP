"""
preprocess.py
-------------
Text preprocessing pipeline for the Emotion Detection project.

Steps applied:
    1. Lowercase conversion
    2. Removal of special characters / punctuation
    3. Tokenization
    4. Stopword removal
    5. Lemmatization
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (safe to call multiple times)
def download_nltk_resources():
    """Download all NLTK resources required by this module."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_resources()

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw text string.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned, tokenised, lemmatised text joined back as a string.

    Examples
    --------
    >>> preprocess_text("I am feeling very happy today!!!")
    'feeling happy today'
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove special characters (keep only alphabets and whitespace)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    tokens = [word for word in tokens if word not in _stop_words]

    # 5. Lemmatize
    tokens = [_lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I got selected for the internship! I'm so excited!",
        "I lost my wallet and I feel terrible.",
        "I am absolutely terrified about the exam results.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Processed: {preprocess_text(s)}")
        print()
