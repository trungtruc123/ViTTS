import os
import unittest

from tests import get_test_output_path
from vitts.utils.config import load_config
from vitts.components.vitts.models import setup_model
from vitts.utils.io import save_checkpoint
from vitts.inference.synthesizer import Synthesizer


class SynthesizerTest(unittest.TestCase):
    # pylint: disable=R0201
    def _create_random_model(self):
        # pylint: disable=global-statement
        config = load_config(os.path.join(get_test_output_path(), "dummy_model_config.json"))
        model = setup_model(config)
        output_path = os.path.join(get_test_output_path())
        save_checkpoint(config, model, None, None, 10, 1, output_path)

    def test_in_out(self):
        self._create_random_model()
        tts_root_path = get_test_output_path()
        tts_checkpoint = os.path.join(tts_root_path, "checkpoint_10.pth")
        tts_config = os.path.join(tts_root_path, "dummy_model_config.json")
        synthesizer = Synthesizer(tts_checkpoint, tts_config, None, None)
        synthesizer.tts("Xin chào, mình tên là trực")
