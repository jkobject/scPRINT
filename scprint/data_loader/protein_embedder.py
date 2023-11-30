import os
import time
import subprocess
import pandas as pd
from torch import load

# https://github.com/agemagician/ProtTrans
# https://academic.oup.com/nargab/article/4/1/lqac012/6534363


class PROTBERT:
    def __init__(
        config="esm-extract",
        pretrained_model="esm2_t33_650M_UR50D",
    ):
        self.config = config
        self.pretrained_model = pretrained_model

    def __call__(self, input_file, output_folder="/tmp/esm_out/", cache=True):
        if not os.path.exists(output_folder) or not cache:
            os.makedirs(output_folder)
            cmd = (
                self.config
                + " "
                + self.pretrained_model
                + " "
                + input_file
                + " "
                + output_folder
            )
            try:
                subprocess.Popen(cmd, shell=True).wait()
            except Exception as e:
                raise RuntimeError(
                    "An error occurred while running the esm-extract command: " + str(e)
                )
        return self.read_results(output_folder)

    def read_results(self, output_folder):
        """
        read_results read multiple .pt files in a folder and convert in a dataframe

        Args:
            output_folder (_type_): _description_
        """
        files = os.listdir(output_folder)
        files = [i for i in files if i.endswith(".pt")]
        results = []
        for file in files:
            results.append(load(output_folder + file))
        return pd.DataFrame(data=results, index=[file.split(".")[0] for file in files])
