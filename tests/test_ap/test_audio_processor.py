import os
import unittest

from tests import get_test_input_path, get_test_output_path, get_test_path
from vitts.utils.config.config_base import BaseAudioConfig
from vitts.utils.audio import AudioProcessor

TEST_PATH = get_test_path()
OUT_PATH = os.path.join(get_test_output_path(), "audio_tests")
WAVE_FILE = os.path.join(get_test_input_path(), "example_1.wav")

os.makedirs(OUT_PATH, exist_ok=True)
conf = BaseAudioConfig(mel_fmax=8000)


class TestAudio(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ap = AudioProcessor(**conf)

    def test_audio_synthesis(self):
        """
        1. load wav
        2. set normalization parameters
        3. extract mel-spec
        4. invert to wav and save output
        """
        print(" Check the process wav -> mel -> wav")

        def _test(max_norm, signal_norm, symmetric_norm, clip_norm):
            self.ap.max_norm = max_norm
            self.ap.signal_norm = signal_norm
            self.ap.symmetric_norm = symmetric_norm
            self.ap.clip_norm = clip_norm

            wav = self.ap.load_wav(WAVE_FILE)
            mel = self.ap.melspectrogram(wav)
            wav_ = self.ap.inv_melspectrogram(mel)
            file_name = "/audio_test_melspec_max{}_signal{}_symmetric{}_clip{}.wav".format(
                max_norm, signal_norm, symmetric_norm, clip_norm
            )
            print("Create file wav at:", file_name)
            self.ap.save_wav(wav_, OUT_PATH + file_name)

        # maxnorm = 1.0
        _test(1.0, False, False, False)
        _test(1.0, True, False, False)
        _test(1.0, True, True, False)
        _test(1.0, True, False, True)
        _test(1.0, True, True, True)
        # maxnorm = 4.0
        _test(4.0, False, False, False)
        _test(4.0, True, False, False)
        _test(4.0, True, True, False)
        _test(4.0, True, False, True)
        _test(4.0, True, True, True)

    def test_normalize(self):
        """
        Check normalization and denormalization
        :return:
        """
        print(" >>>> Begin test normalization and denormalization")
        wav = self.ap.load_wav(WAVE_FILE)
        wav = self.ap.sound_norm(wav)  # normalize audio to get better normalization range below
        self.ap.signal_norm = False
        x = self.ap.melspectrogram(wav)
        x_old = x

        self.ap.signal_norm = True
        self.ap.symmetric_norm = False
        self.ap.clip_norm = False
        self.ap.max_norm = 4.0
        x_norm = self.ap.normalize(x)
        print(
            f" > MaxNorm: {self.ap.max_norm}, ClipNorm:{self.ap.clip_norm}, SymmetricNorm:{self.ap.symmetric_norm}, SignalNorm:{self.ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
        )
        assert (x_old - x).sum() == 0
        # check value range
        assert x_norm.max() <= self.ap.max_norm + 1, f"! Error {x_norm.max()}"
        assert x_norm.min() >= 0 - 1, f"! Error {x_norm.min()}"
        # check denorm
        x_ = self.ap.denormalize(x_norm)
        assert (x - x_).sum() < 1e-3, (x - x_).mean()

        self.ap.signal_norm = True
        self.ap.symmetric_norm = False
        self.ap.clip_norm = True
        self.ap.max_norm = 4.0
        x_norm = self.ap.normalize(x)
        print(
            f" > MaxNorm: {self.ap.max_norm}, ClipNorm:{self.ap.clip_norm}, SymmetricNorm:{self.ap.symmetric_norm}, SignalNorm:{self.ap.signal_norm} Range-> {x_norm.max()} --  {x_norm.min()}"
        )
        assert (x_old - x).sum() == 0
        # check value range
        assert x_norm.max() <= self.ap.max_norm, x_norm.max()
        assert x_norm.min() >= 0, x_norm.min()
        # check denorm.
        x_ = self.ap.denormalize(x_norm)
        assert (x - x_).sum() < 1e-3, (x - x_).mean()

        assert (x_old - x).sum() == 0
        # check value range
        assert x_norm.max() <= self.ap.max_norm, x_norm.max()
        assert x_norm.min() >= 0, x_norm.min()
        # check denorm.
        x_ = self.ap.denormalize(x_norm)
        assert (x - x_).sum() < 1e-3, (x - x_).mean()

    def test_scaler(self):
        scaler_stats_path = os.path.join(get_test_input_path(), "scale_stats.npy")
        conf.stats_path = scaler_stats_path
        conf.preemphasis = 0.0
        conf.do_trim_silence = True
        conf.signal_norm = True

        ap = AudioProcessor(**conf)
        mel_mean, mel_std, linear_mean, linear_std, _= ap.load_stats(scaler_stats_path)
        ap.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)

        self.ap.signal_norm = False
        self.ap.preemphasis = 0.0

        # test scaler forward and backward transforms
        wav = self.ap.load_wav(WAVE_FILE)
        mel_reference = self.ap.melspectrogram(wav)
        mel_norm = ap.melspectrogram(wav)
        mel_denorm = ap.denormalize(mel_norm)
        assert abs(mel_reference - mel_denorm).max() < 1e-4

    def test_compute_f0(self):
        ap = AudioProcessor(**conf)
        wav = ap.load_wav(WAVE_FILE)
        pitch = ap.compute_f0(wav)
        mel = ap.melspectrogram(wav)
        assert pitch.shape[0] == mel.shape[1]