from dataclasses import replace, dataclass
from typing import Dict

# DEFAUT SET OF GRAPHEMES

_pad = "<PAD>"
_eos = "<EOS>"
_bos = "<BOS>"
_blank = "<BLNK>"  # check if we need this alongside with PAD
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĂÂÊÔƯ"
_punctuations = "!'(),-.:;? "
_phonemes = "ÁẠÃẢÀẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆÓÒỎÕỌỐỒỔỖỘÚÙỦŨỤỨỪỬỮỰáạãảàắằẳẵặấầẩẫậéèẻẽẹếềểễệóòỏõọốồổỗộúùủũụứừửữự"


def parse_symbol():
    return {
        "pad": _pad,
        "eos": _eos,
        "bos": _bos,
        "characters": _characters,
        "punctuations": _punctuations,
        "phonemes": _phonemes,
    }


if __name__ == "__main__":
    @dataclass(frozen=True)
    class C:
        x: int
        y: int


    c = C(1, 2)
    c1 = replace(c, x=3)
    # c1 = C(3,2)
    assert c1.x == 3 and c1.y == 2
    _phome_lowers = _phonemes.lower()
    print(_phome_lowers)
