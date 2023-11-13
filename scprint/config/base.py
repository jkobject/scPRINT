import hydra
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

# hydra-torch structured config imports
from hydra_configs.torch.optim import AdadeltaConf
from hydra_configs.torch.optim.lr_scheduler import StepLRConf


@dataclass
class MNISTConf:
    batch_size: int = 64
    test_batch_size: int = 1000
    epochs: int = 14
    no_cuda: bool = False
    dry_run: bool = False
    seed: int = 1
    log_interval: int = 10
    save_model: bool = False
    checkpoint_name: str = "unnamed.pt"
    adadelta: AdadeltaConf = AdadeltaConf()
    steplr: StepLRConf = StepLRConf(
        step_size=1
    )  # we pass a default for step_size since it is required, but missing a default in PyTorch (and consequently in hydra-torch)

@dataclass
class BaseConfig:
    HTFTARGET = "http://bioinfo.life.hust.edu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
    TFLINK = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"

    
cs = ConfigStore.instance()
cs.store(name="mnistconf", node=MNISTConf)