from TTS.tts.utils.text.vietnamese.number_norm import normalize_numbers
from TTS.tts.utils.text.vietnamese.time_norm import normalize_time

def vietnamese_text_to_phonemes(text: str) -> str:
    """
    Convert Vietnamese text to phonemes
    Args:
        text:

    Returns:

    """
    text = normalize_time(text)
    text = normalize_numbers(text)
    return text
