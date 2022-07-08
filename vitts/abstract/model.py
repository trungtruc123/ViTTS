from abc import abstractmethod
from typing import Dict

import torch
from vitts.utils.config.config_parser import ConfigParser
from trainer import TrainerModel


class BaseTrainerModel(TrainerModel):
    """
    Expanding TrainerModel
    inherit it
    """

    @staticmethod
    @abstractmethod
    def init_for_training(config: ConfigParser) -> None:
        ...

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input={}) -> Dict:
        """
        Forward pass for inference.
        It must return a dictionary with the main model output and all the auxiliary outputs. The key ```model_outputs```
        is considered to be the main output and you can add any other auxiliary outputs as you want.

        We don't use `*kwargs` since it is problematic with the TorchScript API.
        Args:
            input (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
        outputs_dict = {"model_outputs": None}
        return outputs_dict

    @abstractmethod
    def load_checkpoint(
            self,
            config: ConfigParser,
            checkpoint_path: str,
            eval: bool = False,
            strict: bool = True
    ) -> None:
        """Load a model checkpoint gile and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
            strict (bool, optional): Match all checkpoint keys to model's keys. Defaults to True.
        """
        ...
