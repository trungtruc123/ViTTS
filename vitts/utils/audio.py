from typing import Dict, Tuple
import librosa
import numpy as np
import scipy.io.wavfile
import scipy.signal
import soundfile as sd
import torch
from torch import nn
from vitts.utils.config.config_parser import ConfigParser


class AudioProcessor(object):
    """
    Audio Processor for ViTTS used by all the data pipline
    TODO: Make this dataclass to replace "BaseAudioConfig" .

    Note:
    All the class arguments are set to default values to enable a flexible initialization
    of the class with the model config. They are not meaningful for all the arguments.

    Abbreviation:

        DB : decibels (đơn vị đo lường cường độ âm thanh)
        amp: amplitude ( biên độ)
        stft: Short time Fourier transform (docs: https://librosa.org/doc/main/generated/librosa.stft.html)
            + hop_length: số lượng mẫu âm thanh giữa các cột STFT liền kề. Default: win_length//4
            + win_length: mỗi khung âm thanh được đánh dấu theo window độ dài và win_length

    Args:

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        resample (bool, optional):
            enable/disable resampling of the audio clips when the target sampling rate does not match the original sampling rate. Defaults to False.

        num_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        log_func (int, optional):
            log exponent used for converting spectrogram aplitude to DB(decibels: đơn vị đo lường cường độ âm thanh).

        min_level_db (int, optional):
            minimum db threshold for the computed melspectrograms. Defaults to None.

        frame_shift_ms (int, optional):
            milliseconds of frames between STFT columns. Defaults to None.

        frame_length_ms (int, optional):
            milliseconds of STFT window length. Defaults to None.

        hop_length (int, optional):
            number of frames between STFT columns. Used if ```frame_shift_ms``` is None. Defaults to None.

        win_length (int, optional):
            STFT window length. Used if ```frame_length_ms``` is None. Defaults to None.

        ref_level_db (int, optional):
            reference DB level to avoid background noise. In general <20DB corresponds to the air noise. Defaults to None.

        fft_size (int, optional):
            FFT window size for STFT. Defaults to 1024.

        power (int, optional):
            Exponent value applied to the spectrogram before GriffinLim. Defaults to None.

        preemphasis (float, optional):
            Preemphasis coefficient. Preemphasis is disabled if == 0.0. Defaults to 0.0.

        signal_norm (bool, optional):
            enable/disable signal normalization. Defaults to None.

        symmetric_norm (bool, optional):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else [0, k], Defaults to None.

        max_norm (float, optional):
            ```k``` defining the normalization range. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        pitch_fmin (int, optional):
            minimum filter frequency for computing pitch. Defaults to None.

        pitch_fmax (int, optional):
            maximum filter frequency for computing pitch. Defaults to None.

        spec_gain (int, optional):
            gain applied when converting amplitude to DB. Defaults to 20.

        stft_pad_mode (str, optional):
            Padding mode for STFT. Defaults to 'reflect'.

        clip_norm (bool, optional):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        griffin_lim_iters (int, optional):
            Number of GriffinLim iterations. Defaults to None.

        do_trim_silence (bool, optional):
            enable/disable silence trimming when loading the audio signal. Defaults to False.

        trim_db (int, optional):
            DB threshold used for silence trimming. Defaults to 60.

        do_sound_norm (bool, optional):
            enable/disable signal normalization. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        stats_path (str, optional):
            Path to the computed stats file. Defaults to None.

        verbose (bool, optional):
            enable/disable logging. Defaults to True.

    """

    def __init__(
            self,
            sample_rate=None,
            resample=False,
            num_mels=None,
            log_func="np.log10",
            min_level_db=None,
            frame_shift_ms=None,
            frame_length_ms=None,
            hop_length=None,
            win_length=None,
            ref_level_db=None,
            fft_size=1024,
            power=None,
            preemphasis=0.0,
            signal_norm=None,
            symmetric_norm=None,
            max_norm=None,
            mel_fmin=None,
            mel_fmax=None,
            pitch_fmax=None,
            pitch_fmin=None,
            spec_gain=20,
            stft_pad_mode="reflect",
            clip_norm=True,
            griffin_lim_iters=None,
            do_trim_silence=False,
            trim_db=60,
            do_sound_norm=False,
            do_amp_to_db_linear=True,
            do_amp_to_db_mel=True,
            do_rms_norm=False,
            db_level=None,
            stats_path=None,
            verbose=True,
            **_,
    ):
        # setup class attributed
        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.log_func = log_func
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.do_amp_to_db_linear = do_amp_to_db_linear
        self.do_amp_to_db_mel = do_amp_to_db_mel
        self.do_rms_norm = do_rms_norm
        self.db_level = db_level
        self.stats_path = stats_path
        # setup exp_func for decibels to amplitude conversation ( cường độ âm thanh tới biên độ)
        if log_func == "np.log":
            self.base = np.e
        elif log_func == "np.log10":
            self.base = 10
        else:
            raise ValueError(" Unknown log_func value. !")

        # setup stft parameters
        if hop_length is None:
            # compute stft parameters from given time values
            self.hop_length, self.win_length = self._stft_parameters()
        else:
            # use stft from config file
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, "! min_level of decibels is 0"
        assert (
                self.win_length <= self.fft_size
        ), f" ! win_length cannot be larger than fft_size: {self.win_length} vs {self.fft_size}"
        members = vars(self)
        if verbose:
            print(" > Setting up Audio Processor...")
            for key, value in members.items():
                print(f" |----> {key}:{value}")
        # create spectrogram utils
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        # setup scaler
        if stats_path and signal_norm:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_stats(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None

    @staticmethod
    def init_from_config(config: ConfigParser, verbose: True):
        if "AUDIO" in config:
            return AudioProcessor(verbose=verbose, **config.get_audio_arguments())
        return AudioProcessor(verbose=verbose, **config.get_audio_arguments())



    def _stft_parameters(self):
        pass

    def _build_mel_basis(self):
        pass

    def load_stats(self, stats_path):
        pass

    def setup_scaler(self, mel_mean, mel_std, linear_mean, linear_std):
        pass
