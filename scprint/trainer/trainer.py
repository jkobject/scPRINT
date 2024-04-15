from lightning.pytorch.callbacks import Callback
from typing import List


class TrainingMode(Callback):
    def __init__(
        self,
        do_denoise: bool = False,
        noise: List[float] = [0.3],
        do_cce: bool = True,
        cce_sim: float = 0.5,  # .6
        cce_scale: float = 0.002,  # .01
        do_ecs: bool = True,
        ecs_threshold: float = 0.3,
        ecs_scale: float = 0.05,  # .1
        do_mvc: bool = False,
        mvc_scale: float = 0.05,
        do_adv_cls: bool = False,
        do_next_tp: bool = False,
        do_generate: bool = False,
        class_scale: float = 0.4,
        optim: str = "adamW",
        mask_ratio: List[float] = [0.3],
        warmup_duration: int = 500,
        weight_decay: float = 0.01,
        fused_adam: bool = True,
        adv_class_scale: float = 0.1,
        lr_reduce_patience: int = 2,
        lr_reduce_factor: float = 0.5,
        do_cls: bool = True,
        do_adv_batch: bool = True,
        run_full_forward: bool = False,
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
        self.fused_adam = fused_adam
        self.mvc_scale = mvc_scale
        self.do_cls = do_cls
        self.adv_class_scale = adv_class_scale
        self.lr_reduce_patience = lr_reduce_patience
        self.lr_reduce_factor = lr_reduce_factor
        self.do_cls = do_cls
        self.do_adv_batch = do_adv_batch
        self.run_full_forward = run_full_forward

    def __repr__(self):
        return (
            f"TrainingMode("
            f"do_denoise={self.do_denoise}, "
            f"noise={self.noise}, "
            f"do_cce={self.do_cce}, "
            f"cce_sim={self.cce_sim}, "
            f"cce_scale={self.cce_scale}, "
            f"do_ecs={self.do_ecs}, "
            f"ecs_threshold={self.ecs_threshold}, "
            f"ecs_scale={self.ecs_scale}, "
            f"do_mvc={self.do_mvc}, "
            f"do_adv_cls={self.do_adv_cls}, "
            f"adv_class_scale={self.adv_class_scale}, "
            f"do_next_tp={self.do_next_tp}, "
            f"do_generate={self.do_generate}, "
            f"class_scale={self.class_scale}, "
            f"mask_ratio={self.mask_ratio}, "
            f"warmup_duration={self.warmup_duration}, "
            f"fused_adam={self.fused_adam}, "
            f"lr_reduce_patience={self.lr_reduce_patience}, "
            f"lr_reduce_factor={self.lr_reduce_factor}, "
            f"mvc_scale={self.mvc_scale}, "
            f"do_cls={self.do_cls}, "
            f"do_adv_batch={self.do_adv_batch}, "
            f"run_full_forward={self.run_full_forward})"
        )

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
        model.fused_adam = self.fused_adam
        model.do_generate = self.do_generate
        model.mvc_scale = self.mvc_scale
        model.do_cls = self.do_cls
        model.adv_class_scale = self.adv_class_scale
        model.lr_reduce_patience = self.lr_reduce_patience
        model.lr_reduce_factor = self.lr_reduce_factor
        model.do_cls = self.do_cls
        model.do_adv_batch = self.do_adv_batch
        model.run_full_forward = self.run_full_forward
