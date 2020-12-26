import unicodedata


def strip_accents(s):
    out=unicodedata.normalize('NFKD', s)
    return "".join(c for c in out if not unicodedata.combining(c))

