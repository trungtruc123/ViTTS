import yaml
import json
import jsbeautifier
from typing import *


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
