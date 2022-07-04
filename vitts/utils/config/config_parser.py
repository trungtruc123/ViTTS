import os
import re
import sys
from typing import *
from vitts.utils.file_util import load_yaml
from vitts.utils.file_util import write_json_beautifier

MAPPING_PACKAGE = {
    "AUDIO": {
        "AudioConfig": "vitts.components.audio"
    },
    "DATASET": {
        "AudioConfig": "vitts.components.audio"
    },
    "TRAINING": {
        "AudioConfig": "vitts.components.audio"
    }
}


class ConfigParser:
    """
    Parser file config (format: yaml or json)
    """

    def __init__(self, configure: Union[Text, Dict]):
        if isinstance(configure, Text) and configure.endswith(".yaml"):
            assert os.path.isfile(configure), f"Configrue file: {configure} not found!!!!"
            self.configure = load_yaml(configure)
        elif isinstance(configure, Dict):
            self.configure = configure
        else:
            raise Exception(f"We do not support {type(configure)},"
                            f"supported: .json and .yaml or dictionary")

    def get_audio_arguments(self) -> Dict:
        assert "AUDIO" in self.configure, f"AUDIO not in config, please check again"
        return self.configure.get("AUDIO", {})

    def get_dataset_arguments(self) -> Dict:
        assert "DATASET" in self.configure, f"DATASET not in config, please check again"
        return self.configure.get("DATASET", {})

    def get_training_arguments(self) -> Dict:
        assert "TRAINING" in self.configure, f"TRAINING not in config, please check again"
        return self.configure.get("TRAINING", {})


def check_argument(
        name,
        c,
        is_path: bool = False,
        prerequest: str = None,
        enum_list: list = None,
        max_val: float = None,
        min_val: float = None,
        restricted: bool = False,
        alternative: str = None,
        allow_none: bool = True,
) -> None:
    """

    :param name: (str): name of the field to be checked
    :param c: (dict) config dictionary
    :param is_path: (bool, optional) if == True check if the path is exist. Default False
    :param prerequest: (list or str, optional): a list of field name that are prerequestedby the target field name.
        Defaults to ```[]```
    :param enum_list: (list, optional): list of possible values for the target field. Defaults to None.
    :param max_val: (float, optional): maximum possible value for the target field. Default None
    :param min_val: (float, optional): minimum possible value for the target field. Default None
    :param restricted:(bool, optional): if ```True``` the target field has to be defined. Defaults to False.
    :param alternative: (str, optional): a field name superceding the target field. Defaults to None.
    :param allow_none: (bool, optional): if ```True``` allow the target field to be ```None```. Defaults to False.
    :return:

    Example:
        >>> num_mels = 5
        >>> check_argument('num_mels', c, restricted=True, min_val=10, max_val=2056)
        >>> fft_size = 128
        >>> check_argument('fft_size', c, restricted=True, min_val=128, max_val=4058)
    """
    # check if None allowed
    if allow_none and c[name] is None:
        return
    if not allow_none:
        assert c[name] is not None, f"! None value is not allowed for {name}"
    # check if restricted and it is check if it exists
    if isinstance(restricted, bool) and restricted:
        assert name in c.keys(), f"! {name} not defined in config.json"
    # check prerequest fields are defined
    if isinstance(prerequest, list):
        assert any(
            f not in c.keys() for f in prerequest
        ), f"! prerequested fields {prerequest} for {name} are not defined."
    else:
        assert (
                prerequest is None or prerequest in c.keys()
        ), f"! prerequested fields {prerequest} for {name}  are not defined."
    # check if the path exists
    if is_path:
        assert os.path.exists(c[name]), f"! path for {name}: {c[name]} does not exist."
    # skip the rest if the alternative field is defined
    if alternative in c.keys() and c[alternative] is not None:
        return
    # check value constraints
    if name in c.keys():
        if max_val is not None:
            assert c[name] <= max_val, f"! {name} is larger than max value {max_val}"
        if min_val is not None:
            assert c[name] >= min_val, f"! {name} is smaller than min value {min_val}"
        if enum_list is not None:
            assert c[name].lower() in enum_list, f"! {name} is not a valid value."


if __name__ == "__main__":
    cofigure = ConfigParser("/home/truc/Documents/ViTTS/config/config.yaml")
    out = cofigure.get_audio_arguments()
    print(out)
