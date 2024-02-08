from . import BaseExperiment

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
import torch
import copy
import gc
import matplotlib.pyplot as plt
import time
import os
from typing import Union

from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

from scprint.dataset.gene_tokenizer import GeneVocab

from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.trainer import prepare_data, prepare_dataloader
from scgpt.trainer import (
    train,
    evaluate,
    eval_testdata as scgpt_train,
    scgpt_evaluate,
    scgpt_test,
)
from scgpt.preprocess import Preprocessor
from scgpt.model import TransformerModel
from scgpt.utils import eval_scib_metrics, load_pretrained
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)


class scGExperiment(BaseExperiment):
    def __init__(
        batch_keys: list[str] = [
            "self_reported_ethnicity_ontology_term_id",
            "assay_ontology_term_id",
        ],
        epochs: int = 5,
        special_tokens: list[str] = ["<pad>", "<unk>", "<mask>"],
        n_hvg=2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_keys = batch_keys
        self.epochs = epochs
        self.special_tokens = special_tokens
        self.n_hvg = n_hvg

    def init_datamodule(self, dataset: str, vocab: Union[str, Vocab] = None):
        if type(dataset) is str:
            self.dataset = sc.read_h5ad(dataset)

        if type(vocab) is str:
            vocab = GeneVocab.from_file(vocab)
        elif vocab is None:
            self.vocab = Vocab(
                VocabPybind(self.dataset["gene_name"] + self.special_tokens, None)
            )  # bidirectional lookup [gene <-> int]
        self.vocab.set_default_index(self.vocab["<pad>"])
        self.dataset.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in self.dataset.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(self.dataset.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        self.only_loc = self.dataset.var["id_in_vocab"] >= 0
        self.dataset = self.dataset[self.only_loc]
        self.dataset.var["gene_ids"] = self.vocab(self.dataset.var["gene_name"])

        self.dataset.obs["batch_id"] = self.dataset.obs[self.batch_keys].apply(
            "_".join, axis=1
        )
        batch_ids = self.dataset.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

    def init_preprocessor(self, data_is_raw=True, filter_gene_by_counts=3, n_bins=60):
        super().init_preprocessor()
        self.preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=self.n_hvg,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )

    def prepare_dataset(self, test_size=0.1):
        # make sure count from 0
        self.preprocessor(self.dataset)
        all_counts = (
            self.dataset.layers["X_binned"]
            if issparse(self.dataset.layers["X_binned"])
            else self.dataset.layers["X_binned"]
        )
        (
            train_data,
            valid_data,
            train_celltype_labels,
            valid_celltype_labels,
            train_batch_labels,
            valid_batch_labels,
        ) = train_test_split(
            all_counts,
            self.dataset.obs["celltype"].values,
            self.dataset.obs["batch_id"].values,
            test_size=test_size,
            shuffle=True,
        )
        tokenized_train = tokenize_and_pad_batch(
            train_data,
            self.dataset.var["gene_ids"].values,
            max_len=self.n_hvg + 1,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=self.config.pad_value,
            append_cls=True,  # append <cls> token at the beginning
            include_zero_gene=True,
        )
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            self.dataset.var["gene_ids"].values,
            max_len=self.n_hvg + 1,
            vocab=self.vocab,
            pad_token=self.config.pad_token,
            pad_value=self.config.pad_value,
            append_cls=True,
            include_zero_gene=True,
        )
        self.logger.log(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        self.logger.log(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )
        train_data_pt, valid_data_pt = prepare_data(
            tokenized_train,
            tokenized_valid,
            train_batch_labels,
            valid_batch_labels,
            self.config,
            self.epoch,
            train_celltype_labels=train_celltype_labels,
            valid_celltype_labels=valid_celltype_labels,
            sort_seq_batch=self.config.per_seq_batch_sample,
        )

        self.train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=self.config.batch_size,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        self.valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=self.config.batch_size,
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

    def init_model(self, model_path=None):
        self.model = TransformerModel(
            len(self.vocab),  # n_tokens
            # TODO:
            self.config.embsize,
            self.config.nhead,
            self.config.d_hid,
            self.config.nlayers,
            vocab=self.vocab,
            dropout=self.config.dropout,
            pad_token=self.config.pad_token,
            pad_value=self.config.pad_value,
            do_mvc=self.config.GEPC,
            do_dab=True,
            use_batch_labels=True,
            num_batch_labels=self.config.num_batch_types,
            domain_spec_batchnorm=self.config.DSBN,
            n_input_bins=self.config.n_input_bins,
            ecs_threshold=self.config.ecs_thres,
            explicit_zero_prob=self.config.explicit_zero_prob,
            use_fast_transformer=self.config.fast_transformer,
            pre_norm=self.config.pre_norm,
        )
        if model_path is not None:
            load_pretrained(self.model, torch.load(model_path), verbose=False)
            # model_config_file = model_dir / "args.json"
            # model_file = model_dir / "best_model.pt"
        self.model.to(self.device)
        self.logger.watch(self.model)

    def init_trainer(self, lr=1e-4, amp=5, schedule_ratio=64, dropout=0.2):
        # TODO: init wandb
        self.criterion = masked_mse_loss
        self.criterion_dab = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            eps=1e-4 if amp else 1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=schedule_ratio
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

    def train(self):
        scgpt_train(
            self.model,
            self.train_loader,
            self.vocab,
            criterion_gep_gepc,
            self.criterion_dab,
            criterion_cls,
            self.scaler,
            self.optimizer,
            self.scheduler,
            self.device,
            self.config,
            self.logger,
            self.epoch,
        )

    def test(self):
        return scgpt_test(
            self.model, adata_t, self.gene_ids, self.vocab, self.config, self.logger
        )
        # TODO:

    def validate(self):
        return scgpt_evaluate(
            self.model,
            self.valid_loader,
            self.vocab,
            criterion_gep_gepc,
            criterion_dab,
            criterion_cls,
            # TODO:
            self.device,
            self.config,
            self.epoch,
            self.logger,
        )

    def fine_tune(self):
        best_val_loss = float("inf")
        best_model = None

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()
            self.train()
            val_loss, val_mre = self.validate()
            elapsed = time.time() - epoch_start_time
            self.logger.log("-" * 89)
            self.logger.log(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
            )
            self.logger.log("-" * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)
                best_model_epoch = epoch
                self.logger.log(f"Best model with score {best_val_loss:5.4f}")

        self.logger.log(f"Saving model to {self.save_folder}")
        torch.save(
            best_model.state_dict(),
            self.save_folder / f"model_e{best_model_epoch}.pt",
        )

        # eval on testdata
        results = self.test(
            adata_t=new_adata,
            include_types=["cls"],
        )
        # self.logger.log_image(results["test/batch_umap"])
        # self.logger.log_image(results["test/celltype_umap"])
        self.logger.log("\n".join(["test/" + k + ": " + v for k, v in results.items()]))
        # TODO:
        scheduler.step()

    def cleanup(self):
        self.logger.wandb.use_artifact(
            self.save_folder + "/best_model.pt", type="model"
        )
        gc.collect()

    def load_experiment(self, experiment_folder):
        checkpoint = torch.load(self._experiment_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # TODO:

    def save_experiment(self):
        # TODO:
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "vocab": self.vocab,
                "config": self.model_config,
            },
            self._experiment_file,
        )
