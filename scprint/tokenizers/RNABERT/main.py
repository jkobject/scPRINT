import torch
from .utils import (
    get_config,
    BertModel,
    BertForMaskedLM,
)
from .dataload import DATA
import numpy as np
import os
import pandas as pd

FILE_LOC = os.path.dirname(os.path.realpath(__file__))
# https://github.com/mana438/RNABERT
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class RNABERT:
    def __init__(
        self,
        batch_size=20,
        config=FILE_LOC + "/RNA_bert_config.json",
        pretrained_model=FILE_LOC + "/bert_mul_2.pth",
        device="cuda",
    ):
        self.file_location = os.path.dirname(os.path.realpath(__file__))
        self.batch_size = batch_size
        self.config = get_config(file_path=config)
        self.max_length = self.config.max_position_embeddings
        self.maskrate = 0
        self.mag = 1

        self.config.hidden_size = self.config.num_attention_heads * self.config.multiple
        model = BertModel(self.config)
        self.model = BertForMaskedLM(self.config, model)
        self.model.to(device)
        print("device: ", device)
        self.device = device
        if device == "cuda":
            self.model = torch.nn.DataParallel(self.model)  # make parallel
        self.model.load_state_dict(torch.load(pretrained_model))

    def __call__(self, fasta_file):
        print("-----start-------")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        data = DATA(config=self.config, batch=self.batch_size)
        seqs, names, dataloader = data.load_data_EMB([fasta_file])
        features = self.make_feature(self.model, dataloader, seqs)
        features = np.array([np.array(embedding).sum(0) for embedding in features])

        return pd.DataFrame(features, index=names)

    def make_feature(self, model, dataloader, seqs):
        model.eval()
        torch.backends.cudnn.benchmark = True
        encoding = []
        for batch in dataloader:
            data, label, seq_len = batch
            inputs = data.to(self.device)
            _, _, encoded_layers = model(inputs)
            encoding.append(encoded_layers.cpu().detach().numpy())
        encoding = np.concatenate(encoding, 0)

        embedding = []
        for e, seq in zip(encoding, seqs):
            embedding.append(e[: len(seq)].tolist())

        return embedding
