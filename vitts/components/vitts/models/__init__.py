from typing import Dict, List, Union

from vitts.utils.generic_utils import find_module


def setup_model(config: "Coqpit", samples: Union[List[List], List[Dict]] = None) -> "BaseTTS":
    print(" > Using model: {}".format(config.model))
    # fetch the right model implementation.
    if "base_model" in config and config["base_model"] is not None:
        MyModel = find_module("vitts.components.vitts.models", config.base_model.lower())
    else:
        MyModel = find_module("vitts.components.vitts.models", config.model.lower())
    model = MyModel.init_from_config(config, samples)
    return model
