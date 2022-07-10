import datetime
import json
import os
import pickle as pickle_tts
import shutil
from typing import Any, Callable, Dict, Union
import fsspec
import torch

from vitts.utils.config.config_parser import ConfigParser


def load_fsspec(
        path: str,
        map_location: Union[
            str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
        **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

        Args:
            path: Any path or url supported by fsspec.
            map_location: torch.device or str.
            **kwargs: Keyword arguments forwarded to torch.load.

        Returns:
            Object stored in path.
    """
    with fsspec.open(path, "rb") as f:
        return torch.load(f, map_location=map_location, **kwargs)


