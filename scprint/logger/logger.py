import logging
import logging.config
from pathlib import Path
from scprint.utils import read_json
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger


class Logger:
    def __init__(self, name, project, directory, do_WandB=False, do_TensorBoard=False):
        if do_WandB:
            self.wandb_logger = WandbLogger(project=project, name=name)
        if do_TensorBoard:
            self.tensor_logger = TensorBoardLogger(directory, name=name)

    def add_config(self, config):
        self.wandb_logger.experiment.config = config

    def save(self):
        pass

    def load(self):
        pass

    def log(self, msg):
        if self.wandb_logger is not None:
            self.wandb_logger.log(msg)
        if self.tensor_logger is not None:
            self.tensor_logger.log(msg)
        print(msg)

    def info(self, msg):
        self.log(msg)

    def warning(self, msg):
        self.log(msg)

    def watch(self, model):
        if self.wandb_logger is not None:
            self.wandb_logger.watch(model)
        if self.tensor_logger is not None:
            self.tensor_logger.watch(model)

    def finish(self):
        if self.wandb_logger is not None:
            self.wandb_logger.finalize()
        if self.tensor_logger is not None:
            self.tensor_logger.finalize()

    def log_image(self, image, name, folder="/tmp/"):
        if type(image) is plt.Figure:
            image.savefig(
                folder + f"embeddings_batch_umap[cls]_e{name}.png",
                dpi=300,
            )
        if self.wandb_logger is not None:
            self.wandb_logger.log_image(images=image, key=name)
        if self.tensor_logger is not None:
            self.tensor_logger.log_image(image, name)
