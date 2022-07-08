from dataclasses import asdict, dataclass, field
from typing import Dict, List
from vitts.utils.config.config_base import BaseTrainingConfig


@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    """
    Share parameters among all the tts models

    Args:

        audio (BaseAudioConfig):
            Audio processor config object instance.

        use_phonemes (bool):
            enable / disable phoneme use.

        phonemizer (str):
            Name of the phonemizer to use. If set None, the phonemizer will be selected by `phoneme_language`.
            Defaults to None.

        phoneme_language (str):
            Language code for the phonemizer. You can check the list of supported languages by running
            `python TTS/tts/utils/text/phonemizers/__init__.py`. Defaults to None.

        compute_input_seq_cache (bool):
            enable / disable precomputation of the phoneme sequences. At the expense of some delay at the beginning of
            the training, It allows faster data loader time and precise limitation with `max_seq_len` and
            `min_seq_len`.

        text_cleaner (str):
            Name of the text cleaner used for cleaning and formatting transcripts.

        enable_eos_bos_chars (bool):
            enable / disable the use of eos and bos characters.

        test_senteces_file (str):
            Path to a txt file that has sentences used at test time. The file must have a sentence per line.

        phoneme_cache_path (str):
            Path to the output folder caching the computed phonemes for each sample.

        characters (CharactersConfig):
            Instance of a CharactersConfig class.

        batch_group_size (int):
            Size of the batch groups used for bucketing. By default, the dataloader orders samples by the sequence
            length for a more efficient and stable training. If `batch_group_size > 1` then it performs bucketing to
            prevent using the same batches for each epoch.

        loss_masking (bool):
            enable / disable masking loss values against padded segments of samples in a batch.

        sort_by_audio_len (bool):
            If true, dataloder sorts the data by audio length else sorts by the input text length. Defaults to `False`.

        min_text_len (int):
            Minimum length of input text to be used. All shorter samples will be ignored. Defaults to 0.

        max_text_len (int):
            Maximum length of input text to be used. All longer samples will be ignored. Defaults to float("inf").

        min_audio_len (int):
            Minimum length of input audio to be used. All shorter samples will be ignored. Defaults to 0.

        max_audio_len (int):
            Maximum length of input audio to be used. All longer samples will be ignored. The maximum length in the
            dataset defines the VRAM used in the training. Hence, pay attention to this value if you encounter an
            OOM error in training. Defaults to float("inf").

        compute_f0 (int):
            (Not in use yet).

        compute_linear_spec (bool):
            If True data loader computes and returns linear spectrograms alongside the other data.

        precompute_num_workers (int):
            Number of workers to precompute features. Defaults to 0.

        use_noise_augment (bool):
            Augment the input audio with random noise.

        start_by_longest (bool):
            If True, the data loader will start loading the longest batch first. It is useful for checking OOM issues.
            Defaults to False.

        add_blank (bool):
            Add blank characters between each other two characters. It improves performance for some models at expense
            of slower run-time due to the longer input sequence.

        datasets (List[BaseDatasetConfig]):
            List of datasets used for training. If multiple datasets are provided, they are merged and used together
            for training.

        optimizer (str):
            Optimizer used for the training. Set one from `torch.optim.Optimizer` or `TTS.utils.training`.
            Defaults to ``.

        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`

        lr_scheduler (str):
            Learning rate scheduler for the training. Use one from `torch.optim.Scheduler` schedulers or
            `TTS.utils.training`. Defaults to ``.

        lr_scheduler_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"warmup": 4000}`.

        test_sentences (List[str]):
            List of sentences to be used at testing. Defaults to '[]'

        eval_split_max_size (int):
            Number maximum of samples to be used for evaluation in proportion split. Defaults to None (Disabled).

        eval_split_size (float):
            If between 0.0 and 1.0 represents the proportion of the dataset to include in the evaluation set.
            If > 1, represents the absolute number of evaluation samples. Defaults to 0.01 (1%).

        use_speaker_weighted_sampler (bool):
            Enable / Disable the batch balancer by speaker. Defaults to ```False```.

        speaker_weighted_sampler_alpha (float):
            Number that control the influence of the speaker sampler weights. Defaults to ```1.0```.

        use_language_weighted_sampler (bool):
            Enable / Disable the batch balancer by language. Defaults to ```False```.

        language_weighted_sampler_alpha (float):
            Number that control the influence of the language sampler weights. Defaults to ```1.0```.

        use_length_weighted_sampler (bool):
            Enable / Disable the batch balancer by audio length. If enabled the dataset will be divided
            into 10 buckets considering the min and max audio of the dataset. The sampler weights will be
            computed forcing to have the same quantity of data for each bucket in each training batch. Defaults to ```False```.

        length_weighted_sampler_alpha (float):
            Number that control the influence of the length sampler weights. Defaults to ```1.0```.
    """