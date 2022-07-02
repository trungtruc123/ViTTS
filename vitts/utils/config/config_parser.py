import os
import re
import sys
from typing import *
from vitts.utils.file_util import load_yaml

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
