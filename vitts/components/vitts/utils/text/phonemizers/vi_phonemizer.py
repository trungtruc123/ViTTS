# from typing import Dict
# from TTS.tts.utils.text.vietnamese.phonemizer import vietnamese_text_to_phonemes
# from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
# from TTS.tts.utils.text.punctuation import Punctuation
#
#
# class VI_Phonemizer(BasePhonemizer):
#     language = "vi"
#
#     def __init__(
#             self, punctuations=Punctuation.default_puncs(),
#             keep_puncs=True,
#             **kwargs
#     ):
#         super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)
#
#     @staticmethod
#     def name():
#         return "vi_phonemizer"
#
#     def _phonemize(self, text: str, separator: str = "|") -> str:
#         ph = vietnamese_text_to_phonemes(text)
#         if separator is not None or separator != "":
#             return separator.join(ph)
#
#         return ph
#
#     def phonemize(self, text: str, separator="|") -> str:
#         """
#         Custom phonemize for VI
#         Args:
#             text:
#             separator:
#
#         Returns:
#
#         """
#         return self._phonemize(text, separator)
#
#     @staticmethod
#     def supported_languages() -> Dict:
#         return {"vi": "Vietnamese (VietNam)"}
#
#     def version(self) -> str:
#         return "0.0.1"
#
#     def is_available(self) -> bool:
#         return True
#
#
# if __name__ =="__main__":
#     text = "Xin chào anh Trực, bây giờ là 9:30 rồi nhé"
#     e = VI_Phonemizer()
#     print(e.supported_languages())
#     print(e.version())
#     print(e.language)
#     print(e.name())
#     print(e.phonemize(text))