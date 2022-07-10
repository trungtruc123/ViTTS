import os
from dataclasses import dataclass, field

from trainer import Trainer, TrainerArgs
from vitts.utils.config import load_config, register_config
from vitts.components.vocoder.configs.hifigan_config import HifiganConfig
from vitts.utils.audio import AudioProcessor
from vitts.components.vocoder.datasets.preprocess import (
    load_wav_data,
    load_wav_feat_data
)
from vitts.components.vocoder.models import setup_model
from tests import get_test_output_path, get_test_input_path

WORK_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_PATH = os.path.join(WORK_DIR, "tests/data/ljspeech")
OUTPUT_PATH = os.path.join(get_test_output_path(), "train_outputs")


@dataclass
class TrainVocoderArgs(TrainerArgs):
    config = HifiganConfig(
        batch_size=8,
        eval_batch_size=8,
        num_loader_workers=0,
        num_eval_loader_workers=0,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1,
        seq_len=1024,
        eval_split_size=1,
        print_step=1,
        print_eval=True,
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
    )
    config.audio.do_trim_silence = True
    config.audio.trim_db = 60

    config.save_json(os.path.join(get_test_output_path(), "test_vocoder_config.json"))
    config_path = os.path.join(get_test_output_path(), "test_vocoder_config.json")


def main():
    args = TrainVocoderArgs()

    if args.config_path or args.continue_path:
        if args.config_path:
            config = load_config(args.config_path)
        elif args.continue_path:
            config = load_config(os.path.join(args.continue_path, "config.json"))
        else:
            from vitts.utils.config.config_share import BaseTrainingConfig
            config_base = BaseTrainingConfig()
            config = register_config(config_base.model)()

    # load training samples
    if "feature_path" in config and config.feature_path:
        # load pre-computed features
        print(f" >>> Loading features from {config.feature_path}")
        eval_sample, train_sample = load_wav_feat_data(config.data_path, config.feature_path, config.eval_split_size)
    else:
        # laod data raw wav files
        eval_sample, train_sample = load_wav_data(config.data_path, config.eval_split_size)

    # setup audio processor
    ap = AudioProcessor(**config.audio)

if __name__=="__main__":
    main()