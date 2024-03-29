import os
from vitts.utils.config.config_base import BaseDatasetConfig
from vitts.utils.generic_utils import get_cuda

def get_device_id():
    use_cuda, _ = get_cuda()
    if use_cuda:
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "":
            GPU_ID = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
        else:
            GPU_ID = "0"
    else:
        GPU_ID = ""
    return GPU_ID


def get_test_path():
    """Returns the path to the test directory."""
    return os.path.dirname(os.path.realpath(__file__))


def get_test_input_path():
    """Returns the path to the test data directory."""
    return os.path.join(get_test_path(), "input")


def get_test_data_path():
    """Returns the path to the test data directory."""
    return os.path.join(get_test_path(), "data")


def get_test_output_path():
    """Returns the path to the directory for test outputs."""
    return os.path.join(get_test_path(), "output")


def run_cli(command):
    exit_status = os.system(command)
    assert exit_status == 0, f" [!] command `{command}` failed."


def get_test_data_config():
    return BaseDatasetConfig(name="ljspeech", path="tests/data/ljspeech/", meta_file_train="metadata.csv")
