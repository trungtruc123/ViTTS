import re
from typing import Text

from vietnamese.abbreviations import abbreviations_vi
from french.abbreviations import abbreviations_fr
from english.abbreviations import abbreviations_en

# Regular expression matching whitespace
_whitespace_re = re.compile(r"\s+")


def abbreviations_lang(text: Text, lang="vi"):
    # Default: language vietnamese
    _abbreviations = abbreviations_vi
    if lang == "en":
        _abbreviations = abbreviations_en
    elif lang == "fr":
        _abbreviations = abbreviations_fr
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()

# if __name__ == "__main__":
#     out = abbreviations_lang("dn có đẹp ko", lang="vi")
#     print(out)
