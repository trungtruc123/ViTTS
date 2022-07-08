import glob
import os
import random
from multiprocessing import Manager

import numpy as np
import torch
from torch.utils.data import Dataset

class GANDataset(Dataset):
    """
    GAN Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly and returns
    random segments of (audio, feature) couples.
    """
    def __init__(
            self,
            ap,
            items,
            seq_len,
            hop_len,
            pad_short,
            conv_pad = 2,
            return_pairs = False,
            is_training = True,
            return_segments = True,
            use_noise_augment = False,
            use_cache = False,
            verbose = False,
    ):
        super().__init__()
        self.ap = ap
        self.item_list = items
        self.compute_feat = not isinstance(items[0], (tuple, list))
        