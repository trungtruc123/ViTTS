import os

def vispeech(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the ViSpeech meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "vispeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items