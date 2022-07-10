import datetime
import importlib

import matplotlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import fsspec
import torch


def get_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return use_cuda, device


def find_module(model_path: str, module_name: str) -> object:
    """
    Find object of module_name in model_path
    Args:
        model_path: path contain model
        module_name: define name of model can found
    """
    module_name = module_name.lower()
    module = importlib.import_module(model_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Vitts", "vitts")
    return text


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict
