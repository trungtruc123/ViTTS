import importlib
import re

from coqpit import Coqpit


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(config: Coqpit):
    """
    Load model directly from configuration
    """
    if "discriminator_model" in config and "generator_model" in config:
        my_model = importlib.import_module("vitts.components.vocoder.models.gan")
        my_model = getattr(my_model, "GAN")
    else:
        my_model = importlib.import_module("vitts.components.vocoder.models." + config.model.lower())
        if config.model.lower() == "wavernn":
            my_model = getattr(my_model, "WAVERNN")
        elif config.model.lower() == "gan":
            my_model = getattr(my_model, "GAN")
        elif config.model.lower() == "wavegrad":
            my_model = getattr(my_model, "WAVEGRAD")
        else:
            try:
                my_model = getattr(my_model, to_camel(config.model))
            except ModuleNotFoundError as e:
                raise ValueError(f" ! Model {config.model} not exist") from e

        print(" >>> Vocoder model: {}".format(config.model))
        return my_model.init_from_config(config)
