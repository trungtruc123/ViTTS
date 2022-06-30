import re
from typing import Text
from anyascii import anyascii
from vietnamese.abbreviations import abbreviations_vi
from french.abbreviations import abbreviations_fr
from english.abbreviations import abbreviations_en

from english.time_norm import expand_time_english
from english.number_norm import normalize_numbers

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


def lowercase(text: Text):
    return text.lower()


def collapse_whitespace(text: Text):
    return re.sub(_whitespace_re, " ", text).strip()


def convert_to_ascii(text: Text):
    return anyascii(text)


def remove_symbols(text: Text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text


def replace_symbols(text: Text, lang='vi'):
    """
    Function will replace the special symbols
    :param text:
    :param lang: language
    :return:
    """
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    if lang == "vi":
        text = text.replace("&", "và")
    elif lang == "en":
        text = text.replace("&", "and")
    elif lang == "fr":
        text = text.replace("&", "et")
    return text


def basic_clean(text: Text):
    """
    Basic pipeline that lowercase and collapses whitespace
    :param text:
    :return:
    """
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    # text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_time_english(text)
    text = normalize_numbers(text)
    text = abbreviations_lang(text, lang="en")
    text = replace_symbols(text, lang="en")
    text = remove_symbols(text)
    text = collapse_whitespace(text)
    return text


def french_cleaners(text):
    """Pipeline for French text. There is no need to expand numbers, phonemizer already does that"""
    text = abbreviations_lang(text, lang="fr")
    text = lowercase(text)
    text = replace_symbols(text, lang="fr")
    text = remove_symbols(text)
    text = collapse_whitespace(text)
    return text

if __name__ == "__main__":
    text = "what time is it : 10:30"
    # out = abbreviations_lang("dn có đẹp ko", lang="vi")
    out = english_cleaners(text)
    print(out)
