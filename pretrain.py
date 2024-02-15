from lightning.pytorch import Trainer, seed_everything

from scprint import scPrint
from scprint.cli import MyCLI
from scdataloader import DataModule

import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.tuner import Tuner


from lightning.pytorch.cli import ArgsType


torch.set_float32_matmul_precision("medium")
seed_everything(seed, workers=True)


def main(args: ArgsType = None):
    cli = MyCLI(args=args)

    if logger == "wandb":
        wandb_logger = WandbLogger(project=project, save_dir=logdir)
    # TODO: for each pretraining, make a commit of the code under a tag logger.experiment.name in a specific branch
    # if do_commit:
    #    import os
    #    tag = wandb_logger.experiment.name if logger == "wandb" else "my_model"
    #    os.system(f"git add . && git commit -m '{tag}'")

    datamodule = DataModule(
        collection_name=collection_name,
        organisms=organisms,
        obs=all_labels,
        clss_to_pred=labels_weighted_sampling,
        hierarchical_clss=hierarchical_labels,
        label_to_weight=labels_weighted_sampling,
        gene_embeddings=gene_embeddings,
        validation_split=0.2,
        test_split=0,
        max_len=1000,
        add_zero_genes=100,
        batch_size=64,
        num_workers=8,
    )
    # TODO: manage gene pos
    datamodule.setup()
    for i in datamodule.train_dataloader():
        break

    model = scPrint(
        genes=df.index.tolist(),
        d_model=d_model,
        nhead=4,
        d_hid=d_model,
        nlayers=4,
        layers_cls=[],
        labels=labels,
        cls_hierarchy=cls_hierarchies,
        dropout=0.2,
        transformer="flash",
        precpt_gene_emb=gene_embeddings,
        gene_pos_enc=df["pos"].tolist(),
        mvc_decoder="inner product",
        label_decoders=decoders,
        pred_embedding=pred_embedding,
    )

    wandb_logger.watch(model, log_graph=True)

    # create a function to transform an scGPT checkpoint to an scPrint's
    # ckpt = torch.load("../../scGPT/save/model_e6.pt")
    # scPrint.load_from_checkpoint("../../scGPT/save/model_e6.pt")

    # tlogger = TensorBoardLogger(save_dir="../data/tensorboard")

    # pytorch_prof = PyTorchProfiler(
    #    "./data/tensorboard",
    #    emit_nvtx=False,
    #    group_by_input_shape=True,
    #    record_shapes=True,
    #    profile_memory=True,
    #    with_stack=True,
    #    on_trace_ready=torch.profiler.tensorboard_trace_handler("./data/tensorboard/"),
    # )
    # sets seeds for numpy, torch and python.random.

    st_weights = StochasticWeightAveraging(swa_lrs=1e-2)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=-1)

    trainer = Trainer(
        precision=16,
        profiler="simple",
        callbacks=[checkpoint_callback, st_weights],
        gradient_clip_val=10,
        max_time=max_time,
        logger=wandb_logger,
    )

    # tuner = Tuner(trainer)

    # Run learning rate finder
    # lr_finder = tuner.lr_find(model)
    # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # update hparams of the model
    # model.hparams.lr = new_lr

    # model.labels = {}
    # model.expr_decoder.nfirst_labels_to_skip=3
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
