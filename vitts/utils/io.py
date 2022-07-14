import datetime
import json
import os
import pickle as pickle_tts
import shutil
from typing import Any, Callable, Dict, Union
import fsspec
import torch
from coqpit import Coqpit

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


def save_fsspec(state: Any, path: str, **kwargs):
    """Like torch.save but can save to other locations (e.g. s3:// , gs://).

    Args:
        state: State object to save
        path: Any path or url supported by fsspec.
        **kwargs: Keyword arguments forwarded to torch.save.
    """
    with fsspec.open(path, "wb") as f:
        torch.save(state, f, **kwargs)


def save_model(config, model, optimizer, scaler, current_step, epoch, output_path, **kwargs):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if isinstance(optimizer, list):
        optimizer_state = [optim.state_dict() for optim in optimizer]
    elif optimizer.__class__.__name__ == "CapacitronOptimizer":
        optimizer_state = [optimizer.primary_optimizer.state_dict(), optimizer.secondary_optimizer.state_dict()]
    else:
        optimizer_state = optimizer.state_dict() if optimizer is not None else None

    if isinstance(scaler, list):
        scaler_state = [s.state_dict() for s in scaler]
    else:
        scaler_state = scaler.state_dict() if scaler is not None else None

    if isinstance(config, Coqpit):
        config = config.to_dict()

    state = {
        "config": config,
        "model": model_state,
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "step": current_step,
        "epoch": epoch,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    save_fsspec(state, output_path)


def save_checkpoint(
        config,
        model,
        optimizer,
        scaler,
        current_step,
        epoch,
        output_folder,
        **kwargs,
):
    file_name = "checkpoint_{}.pth".format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print("\n > CHECKPOINT : {}".format(checkpoint_path))
    save_model(
        config,
        model,
        optimizer,
        scaler,
        current_step,
        epoch,
        checkpoint_path,
        **kwargs,
    )
