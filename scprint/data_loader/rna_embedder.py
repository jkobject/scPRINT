import random
import time
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, datasets
import copy
from Bio import SeqIO
import argparse
from utils.bert import (
    get_config,
    BertModel,
    set_learned_params,
    BertForMaskedLM,
    visualize_attention,
    show_base_PCA,
    fix_params,
)
from module import Train_Module
from dataload import DATA, MyDataset
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import os
import time
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
)
import torch.nn.functional as F
from sklearn.cluster import (
    MiniBatchKMeans,
    KMeans,
    AgglomerativeClustering,
    SpectralClustering,
)
import itertools

import alignment_C as Aln_C

# https://github.com/mana438/RNABERT
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class RNABERT:
    def __init__(
        batch_size,
        config="./RNA_bert_config.json",
        pretrained_model="./pretrained_model/pytorch_model.bin",
        device="cuda",
    ):
        self.batch_size = batch_size
        self.config = get_config(file_path=config)
        self.config.hidden_size = self.config.num_attention_heads * self.config.multiple
        model = BertModel(config)
        model = BertForMaskedLM(config, model)
        model.to(device)
        print("device: ", device)
        if device == "cuda":
            model = torch.nn.DataParallel(model)  # make parallel
        model.load_state_dict(torch.load(pretrained_model))

    def __call__():
        current_time = datetime.datetime.now()
        print("-----start-------")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        data = DATA(args, config)
        seqs, label, test_dl = data.load_data_EMB(args.data_embedding)
        features = train.make_feature(model, test_dl, seqs)
