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
