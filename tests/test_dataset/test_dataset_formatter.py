import os
import unittest

from tests import get_test_input_path, get_test_data_path
from vitts.components.vitts.datasets.formatters import vispeech


class TestTTSFormatter(unittest.TestCase):

    def test_vispeech_voice_processor(self):
        root_path = os.path.join(get_test_data_path(), "vispeech")
        meta_file = "metadata.csv"
        item = vispeech(root_path, meta_file)
        a = item[0]
        assert item[0]["text"] == "bạn biết đấy đến một giai đoạn nhất định sự lười biếng đến trường của một đứa trẻ là việc mà bất cứ bậc phụ huynh nào cũng phải trải qua\n"
        assert item[0]["audio_file"] == os.path.join(root_path, "wavs", "1/13897.wav")
        assert item[0]["speaker_name"] == "vispeech"