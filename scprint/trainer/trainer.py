from lightning.pytorch.callbacks import Callback


class TrainingMode(Callback):
    def __init__(
        self,
        do_denoise=True,
        noise=[0.3],
        do_cce=True,
        cce_sim=0.5,
        do_ecs=True,
        ecs_threshold: float = 0.3,
        ecs_scale: float = 1.0,
        do_mvc=False,
        do_adv_cls=False,
        do_next_tp=False,
        class_scale: float = 1.0,
        mask_ratio=[0.15, 0.3],
    ):
        super().__init__()
        self.do_denoise = do_denoise
        self.noise = noise
        self.do_cce = do_cce
        self.cce_sim = cce_sim
        self.do_ecs = do_ecs
        self.ecs_threshold = ecs_threshold
        self.ecs_scale = ecs_scale
        self.do_mvc = do_mvc
        self.do_adv_cls = do_adv_cls
        self.do_next_tp = do_next_tp
        self.class_scale = class_scale
        self.mask_ratio = mask_ratio

    def on_fit_start(self, trainer, model):
        # do something with all training_step outputs, for example:
        model.do_denoise = self.do_denoise
        model.noise = self.noise
        model.do_cce = self.do_cce
        model.cce_sim = self.cce_sim
        model.do_ecs = self.do_ecs
        model.ecs_threshold = self.ecs_threshold
        model.ecs_scale = self.ecs_scale
        model.do_mvc = self.do_mvc
        model.do_adv_cls = self.do_adv_cls
        model.do_next_tp = self.do_next_tp
        model.class_scale = self.class_scale
        model.mask_ratio = self.mask_ratio
