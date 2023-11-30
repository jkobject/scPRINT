import torch
from utils.bert import (
    get_config,
    BertModel,
    BertForMaskedLM,
)
from dataload import DATA
import datetime
import time

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
