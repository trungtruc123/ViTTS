import os
import shutil
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests import get_test_data_path, get_test_output_path
from vitts.utils.config.config_share import BaseDatasetConfig, BaseTTSConfig
from vitts.components.vitts.datasets import TTSDataset, load_tts_samples
from vitts.components.vitts.utils.text.tokenizer import TTSTokenizer
from vitts.utils.audio import AudioProcessor

OUTPATH = os.path.join(get_test_output_path(), "load_tests/")
os.makedirs(OUTPATH, exist_ok=True)

# create dummy config for testing data loader
c = BaseTTSConfig(
    text_cleaner="vi_cleaners",
    num_loader_workers=0,
    batch_size=2,
    use_noise_augment=False,
)
c.r = 5
c.data_path = os.path.join(get_test_data_path(), "vispeech")
ok_vispeech = os.path.exists(c.data_path)

dataset_config = BaseDatasetConfig(
    name="vispeech",
    meta_file_train="metadata.csv",
    meta_file_val=None,
    path=c.data_path,
    language="vi"
)

DATA_EXIST = True
if not os.path.exists(c.data_path):
    DATA_EXIST = False

print(" > Dynamic data loader test: {}".format(DATA_EXIST))


class TestTTSDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_loader_iter = 4
        self.ap = AudioProcessor(**c.audio)

    def _create_dataloader(self, batch_size, r, bgs, start_by_longest=False):
        # load dataset
        meta_data_train, meta_data_eval = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_size=0.2,
        )
        items = meta_data_train + meta_data_eval
        tokenizer = TTSTokenizer.init_from_config(c)

        dataset = TTSDataset(
            outputs_per_step=r,
            compute_linear_spec=True,
            return_wav=True,
            tokenizer=tokenizer,
            ap=self.ap,
            samples=items,
            batch_group_size=bgs,
            min_text_len=c.min_text_len,
            max_text_len=c.max_text_len,
            min_audio_len=c.min_audio_len,
            max_audio_len=c.max_audio_len,
            start_by_longest=start_by_longest
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last= True,
            num_workers= c.num_loader_workers
        )
        return  dataloader, dataset

    def test_loader(self):
        if ok_vispeech:
            dataloader, dataset = self._create_dataloader(1,1,0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break

                text_input = data["token_id"]
                _ = data["token_id_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                _ = data["stop_targets"]
                _ = data["item_idxs"]
                wavs = data["waveform"]

                neg_values = text_input[text_input <0]
                check_count = len(neg_values)

                # check basic conditions
                self.assertEqual(check_count, 0)
                self.assertEqual(linear_input.shape[0], mel_input.shape[0], c.batch_size)
                self.assertEqual(linear_input.shape[2], self.ap.fft_size // 2 + 1)
                self.assertEqual(mel_input.shape[2], c.audio["num_mels"])
                self.assertEqual(wavs.shape[1], mel_input.shape[1] * c.audio.hop_length)
                self.assertIsInstance(speaker_name[0], str)
