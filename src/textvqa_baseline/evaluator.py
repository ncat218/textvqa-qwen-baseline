import re
import string

from .io_utils import parse_answers


_CONTRACTIONS = {"cant": "can't", "dont": "don't", "doesnt": "doesn't", "isnt": "isn't",
                 "arent": "aren't", "wasnt": "wasn't", "werent": "weren't", "wont": "won't",
                 "couldnt": "couldn't", "wouldnt": "wouldn't", "shouldnt": "shouldn't",
                 "thats": "that's", "whats": "what's", "theres": "there's", "theyre": "they're",
                 "youre": "you're", "im": "i'm", "ive": "i've", "ill": "i'll"}
_DIGITS = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
           "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_vqa(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"(?<!\d)\.(?!\d)|(?<=\D)\.|\.(?=\D)", " ", text)
    punctuation = string.punctuation.replace("'", "").replace(".", "")
    text = text.translate(str.maketrans({c: " " for c in punctuation}))
    words = []
    for word in text.split():
        if word in {"a", "an", "the"}:
            continue
        word = _DIGITS.get(word, word)
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def score_answer(prediction: str, answers: list[str]) -> float:
    normalized = normalize_vqa(prediction)
    matching = sum(normalize_vqa(answer) == normalized for answer in answers)
    return min(matching / 3.0, 1.0)


def score_row(prediction: dict, reference_row: dict) -> float:
    return score_answer(prediction["normalized_prediction"], parse_answers(reference_row))
