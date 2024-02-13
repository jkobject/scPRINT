"""Entry point for scprint."""


from scprint import scPrint
from scprint.cli import MyCLI
from scdataloader import DataModule

from lightning.pytorch.cli import ArgsType

# torch.set_float32_matmul_precision("medium")


def main(args: ArgsType = None):
    cli = MyCLI(
        scPrint,
        DataModule,
        args=args,
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":  # pragma: no cover
    main()
