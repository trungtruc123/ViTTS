import re
from typing import Dict
from Vin2w import n2w, w2n

_commas_number_re = re.compile(r"([0-9][0-9\,]+[0-9]+)")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"-?[0-9]+")
_currency_re = re.compile(r"(£|\$|¥)([0-9\,\.]*[0-9]+)")


def remove_commas(m):
    return m.group(1).replace(",", "")


def expand_decimal_point(m):
    return m.group(1).replace(".", " chấm ")


def expand_ordinal(m):
    tmp = m.group(0)
    tmp = tmp.replace("st", "")
    tmp = tmp.replace("nd", "")
    tmp = tmp.replace("rd", "")
    tmp = tmp.replace("th", "")
    return n2w(tmp)


def expand_number(m):
    '''
    Convert number to words
    :param m:
    :return:
    '''
    num = m.group(0)
    return n2w(num)


def normalize_numbers(text):
    text = re.sub(_commas_number_re, remove_commas, text)
    text = re.sub(_decimal_number_re, expand_decimal_point, text)
    text = re.sub(_ordinal_re, expand_ordinal, text)
    text = re.sub(_number_re, expand_number, text)
    return text


if __name__ == "__main__":
    from Vin2w.w2n import w2n
    from Vin2w.n2w import n2w

    text = "bạn hiểu được 10,003.5 và vị trí 4st không "
    # text_w = "một nghìn không trăm lẻ bốn"
    # out = n2w(text)
    # # out = w2n(text_w)
    out = normalize_numbers(text)
    print(out)
