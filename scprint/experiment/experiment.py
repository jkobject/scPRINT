from dataclasses import dataclass
import torch
import lamindb as ln
import pandas as pd
import lnschema_bionty as lb
from scprint.dataset import Dataset
from scprint.logger import Logger
from scprint.utils import set_seed
import time
from typing import Union
from pathlib import Path
import wandb

# hydra-torch structured config imports
from hydra.core.config_store import ConfigStore


class BaseExperiment:
    def __init__(
        self,
        name: str,
        description: str,
        dataset_name: str,
        model_config: dict,
        experiment_config: dict,
        epoch: int = 0,
        device: torch.device = torch.device("cpu"),
        save_folder: str = "~/scprint/save/",
        seed: int = 42,
        do_TensorBoard=False,
        do_WandB=False,
    ):
        self.name = name
        self.description = description
        self.device = device
        self.epoch = epoch

        # self.config = ConfigStore.instance()
        # self.config.store(name="experiment", node=experiment_config)
        set_seed(seed)

        self.save_dir = Path(
            f"{save_folder}_{dataset_name}-{time.strftime('%b%d-%H-%M')}/"
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = Logger(
            name,
            "base",
            self.save_dir,
            do_WandB=do_WandB,
            do_TensorBoard=do_TensorBoard,
        )
        self.logger.add_config(self.__dict__)

        self.logger.log("experiment initialized")
        self.logger.log("saved to folder: " + str(self.save_dir))
        self.logger.log(f"save to {self.save_dir}")

        if dataset_name is not None:
            self.init_datamodule(dataset_name)
            self.logger.log(f"dataset: {dataset_name} is loaded")

        self.init_preprocessor()
        if model_config is not None:
            self.init_model(model_config)

    def init_datamodule(self, dataset: Union[ln.Dataset, str]):
        if type(dataset) == str:
            dataset = ln.Dataset(name=dataset).first()
        return False
        # self.dataset = Dataset(dataset, lb=lb,# **self.config)

    def init_preprocessor(self):
        pass

    def init_model(self, model_config):
        # model_config_file = model_dir / "args.json"
        # model_file = model_dir / "best_model.pt"
        pass

    def init_trainer(self):
        pass

    def train(self):
        # TODO: should be started by calling the train function with the logger (lightning)
        # Else do it during init trainer
        # self.run = wandb.init(
        #    config=self.config,
        #    project="scGPT",
        #    reinit=True,
        #    settings=wandb.Settings(start_method="fork"),
        # )
        # self.config = wandb.config
        pass

    def test(self):
        raise NotImplementedError("BaseExperiment.run()")

    def validate(self):
        raise NotImplementedError("BaseExperiment.run()")

    def get_name(self):
        return self.name

    def get_description(self):
        return self.description

    def load(experiment_folder):
        checkpoint = torch.load(experiment_folder)
        return BaseExperiment(
            model=checkpoint["model"],
            model_config=checkpoint["config"],
            optimizer=checkpoint["optimizer"],
            loss=checkpoint["loss"],
            _experiment_file=experiment_folder,
            epoch=checkpoint["epoch"],
        )
        # self.model.load_state_dict(checkpoint["model_state_dict"])

    def save(self):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.model_config,
            },
            self._experiment_file,
        )
