from lightning.pytorch.callbacks import Callback
from typing import List


class TrainingMode(Callback):
    def __init__(
        self,
        do_denoise: bool = True,
        noise: List[float] = [0.3],
        do_cce: bool = True,
        cce_sim: float = 0.5,
        cce_scale: float = 1.0,
        do_ecs: bool = True,
        ecs_threshold: float = 0.3,
        ecs_scale: float = 1.0,
        do_mvc: bool = False,
        do_adv_cls: bool = False,
        do_next_tp: bool = False,
        do_generate: bool = False,
        class_scale: float = 1.0,
        optim: str = "adamW",
        mask_ratio: List[float] = [0.15, 0.3],
        warmup_duration: int = 500,
        weight_decay: float = 0.01,
        fused_adam: bool = False,
        lr_patience: int = 3,
    ):
        """
        TrainingMode a callback to set the training specific info to the model.

        This is because lightning is unfortunately setup this way. the model should be separated from training
        but at the same time it has training specific methods... so we have to do this.

        Args:
            see @model.py
        """
        super().__init__()
        self.do_denoise = do_denoise
        self.noise = noise
        self.do_cce = do_cce
        self.cce_sim = cce_sim
        self.cce_scale = cce_scale
        self.do_ecs = do_ecs
        self.ecs_threshold = ecs_threshold
        self.ecs_scale = ecs_scale
        self.do_mvc = do_mvc
        self.do_adv_cls = do_adv_cls
        self.do_next_tp = do_next_tp
        self.do_generate = do_generate
        self.class_scale = class_scale
        self.mask_ratio = mask_ratio
        self.warmup_duration = warmup_duration
        self.weight_decay = weight_decay
        self.fused_adam = fused_adam
        self.lr_patience = lr_patience
        self.optim = optim

    def on_fit_start(self, trainer, model):
        # do something with all training_step outputs, for example:
        model.do_denoise = self.do_denoise
        model.noise = self.noise
        model.do_cce = self.do_cce
        model.cce_sim = self.cce_sim
        model.cce_scale = self.cce_scale
        model.do_ecs = self.do_ecs
        model.ecs_threshold = self.ecs_threshold
        model.ecs_scale = self.ecs_scale
        model.do_mvc = self.do_mvc
        model.do_adv_cls = self.do_adv_cls
        model.do_next_tp = self.do_next_tp
        model.class_scale = self.class_scale
        model.mask_ratio = self.mask_ratio
        model.warmup_duration = self.warmup_duration
        model.weight_decay = self.weight_decay
        model.fused_adam = self.fused_adam
        model.lr_patience = self.lr_patience
        model.do_generate = self.do_generate
        model.optim = self.optim
