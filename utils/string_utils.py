import unicodedata
from typing import List
from utils.stop_words import STOP_WORDS
import regex as re
import string

punctuation = "".join(c for c in string.punctuation if not c=='\'')

def strip_accents(s):
    out=unicodedata.normalize('NFKD', s)
    return "".join(c for c in out if not unicodedata.combining(c))


def preprocessing(line):
    if not isinstance(line, str):
      return ""
    line = line.lower()
    line = line.replace('\n', '')
    line = re.sub(r"[{}]".format(punctuation), " ", line)
    line = " ".join(t for t in line.split(" ") if t not in STOP_WORDS)
    return strip_accents(line).strip()

NUM_RE = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def remove_numbers(texts:List):
    return list(map(lambda t: '<num>' if NUM_RE.match(t) else t, texts))
