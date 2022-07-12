import unittest

from vitts.components.vitts.utils.text.tokenizer import TTSTokenizer
from vitts.components.vitts.utils.text.characters import Graphemes


class TestTokenizer(unittest.TestCase):
    def test_tokinzer_non_phonemes(self):
        tokenizer = TTSTokenizer(
            use_phonemes=False,
            characters=Graphemes()
        )
        text = "ưTôi tên là trự"
        ids = tokenizer.text_to_ids(text)
        text_hat = tokenizer.ids_to_text(ids)
        assert text_hat == text
