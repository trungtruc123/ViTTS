import os

import fsspec
import yaml
import json
import jsbeautifier
from typing import *
from coqpit import Coqpit
from vitts.utils.generic_utils import find_module


def load_config(config_path: str) -> None:
    """
    Import `json` or `yaml` files as TTS configs. First, load the input file as a `dict` and check the model name
    to find the corresponding Config class. Then initialize the Config.

    Args:
        config_path (str): path to the config file.

    Raises:
        TypeError: given config file has an unknown type.

    Returns:
        Coqpit: TTS config object.
    """
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            raise "! Error not read file json"
    else:
        raise TypeError(f"! Unknown config type. Please use file yaml or json")
    config_dict.update(data)
    model_name = _process_model_name(config_dict)
    config_class = register_config(model_name.lower())
    config = config_class()
    config.from_dict(config_dict)
    return config


def _process_model_name(config_dict: Dict) -> str:
    """Format the model name as expected. It is a band-aid for the old `vocoder` model names.

    Args:
        config_dict (Dict): A dictionary including the config fields.

    Returns:
        str: Formatted modelname.
    """
    model_name = config_dict["model"] if "model" in config_dict else config_dict["generator_model"]
    model_name = model_name.replace("_generator", "").replace("_discriminator", "")
    return model_name


def register_config(model_name: str) -> Coqpit:
    """Find the right config for the given model name.

    Args:
        model_name (str): Model name.

    Raises:
        ModuleNotFoundError: No matching config for the model name.

    Returns:
        Coqpit: config class.
    """
    config_class = None
    config_name = model_name + "_config"
    paths = ["vitts.components.encoder.configs", "vitts.components.vocoder.configs"]
    for path in paths:
        try:
            config_class = find_module(path, config_name)
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError(f"! Config for {model_name} cannot be found.")
    return config_class


def load_yaml(file_path: Text) -> Dict:
    """
    Load the context in yaml file
    :param file_path:
    :return: Dict from file yaml
    """
    with open(file_path, 'r') as file:
        yaml_data = yaml.load(file, Loader=yaml.Loader)
    return yaml_data


def load_json(file_path: Text) -> Union[Dict, List]:
    """
    Load the context in json file
    :param file_path:
    :return: json path
    """
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def write_json_beautifier(file_path: Text, dict_info: Union[Dict, List]) -> None:
    """
    Write the content from dictionary into file with a beautiful format
    :param file_path: file path
    :param dict_info: Dict will be dump
    :return:
    """
    opts = jsbeautifier.default_options()
    opts.indent_size = 2
    dict_ = jsbeautifier.beautify(json.dumps(dict_info, ensure_ascii=False), opts)
    with open(file_path, 'w', 'utf-8') as f:
        f.write(dict_)


if __name__ == "__main__":
    path_yaml = "/home/truc/Documents/ViTTS/config/config.yaml"
    path_json = "/home/truc/Documents/ViTTS/config/config.json"
    a = load_yaml(path_yaml)
    print(a)
    b = load_json(path_json)
    print(b)
