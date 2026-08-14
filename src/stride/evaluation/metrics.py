from __future__ import annotations

import re
import string
from collections import Counter


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SPACE = re.compile(r"\s+")
_PERIOD = re.compile(r"(?<!\d)\.(?!\d)")
_COMMA_NUMBER = re.compile(r"(?<=\d),(?=\d)")
_YES = {"yes", "yeah", "true", "1"}
_NO = {"no", "false", "0"}
EVALUATION_VERSION = "evalai-vqa-v1"
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}
_CONTRACTIONS = {
    "arent": "aren't", "cant": "can't", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "hes": "he's", "im": "i'm", "isnt": "isn't", "itll": "it'll",
    "its": "it's", "ive": "i've", "shouldnt": "shouldn't",
    "wasnt": "wasn't", "werent": "weren't", "wont": "won't",
    "wouldnt": "wouldn't", "youre": "you're", "youve": "you've",
}


def normalize_answer(text: str) -> str:
    """EvalAI/VQA-style answer normalization used by all string metrics."""
    text = str(text).replace("\n", " ").replace("\t", " ").lower().strip()
    text = _COMMA_NUMBER.sub("", text)
    text = text.replace(",", " ")
    text = _PERIOD.sub("", text)
    punctuation = (
        string.punctuation.replace("'", "")
        .replace(":", "")
        .replace(".", "")
        .replace(",", "")
    )
    text = text.translate(
        str.maketrans({character: " " for character in punctuation})
    )
    words = []
    for word in _SPACE.sub(" ", text).strip().split(" "):
        word = _NUMBER_WORDS.get(word, word)
        if word and not _ARTICLES.fullmatch(word):
            words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def extract_choice(text: str) -> str:
    value = str(text).strip().upper()
    direct = re.fullmatch(r"[\(\[]?\s*([A-E])\s*[\)\].,:]?", value)
    if direct:
        return direct.group(1)
    explicit = re.search(
        r"\b(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*"
        r"[\(\[]?([A-E])(?:\b|[\)\].,:])",
        value,
    )
    if explicit:
        return explicit.group(1)
    parenthesized = re.search(
        r"(?:^|\s)[\(\[]([A-E])[\)\]](?:\s|$|[.,:])", value
    )
    return parenthesized.group(1) if parenthesized else ""


def _vqa_accuracy(prediction: str, references: list[str]) -> float:
    """Official leave-one-annotator-out VQA consensus accuracy."""
    if not references:
        return 0.0
    matches = Counter(references)[prediction]
    total = 0.0
    for reference in references:
        other_matches = matches - int(reference == prediction)
        total += min(1.0, other_matches / 3.0)
    return total / len(references)


def score_prediction(prediction: str, answers: tuple[str, ...], metric: str) -> float:
    pred = normalize_answer(prediction)
    refs = [normalize_answer(a) for a in answers]
    if metric == "exact_match":
        return float(pred in refs)
    if metric == "contains":
        return float(any(ref and ref in pred for ref in refs))
    if metric == "vqa_accuracy":
        return _vqa_accuracy(pred, refs)
    if metric == "yes_no":
        pred_label = "yes" if pred.split(" ")[0] in _YES else "no" if pred.split(" ")[0] in _NO else pred
        ref_labels = ["yes" if r in _YES else "no" if r in _NO else r for r in refs]
        return float(pred_label in ref_labels)
    if metric == "multiple_choice":
        return float(extract_choice(prediction) in {extract_choice(a) for a in answers})
    raise KeyError(f"Unknown metric {metric!r}")
