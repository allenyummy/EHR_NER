# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Build a pipleine for data processing including data transformation and data augmentation

import os
import logging
from datetime import datetime
from data_process.dataTransformer import DataTransformer
from data_process.dataAugmentator import DataAugmentator

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    target_dataset = "train"

    # --- Tag2Squad ---
    out_v = "V0"
    dt_v0 = DataTransformer(
        version=out_v, time=datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    )
    dt_v0.tag2squad(
        input_data_path=os.path.join(
            "data", "sl", "0817_8786_concat_num", target_dataset + ".txt"
        ),
        output_data_path=os.path.join("data", "final", out_v, target_dataset + ".json"),
    )

    # --- AugmentByTrainedModel ---
    in_v = "V0.0"
    out_v = "V1.1"
    da_a = DataAugmentator(
        version=out_v,
        time=datetime.today().strftime("%Y/%m/%d_%H:%M:%S"),
        query_path=os.path.join("data", "final", "query", "simqasl_query.json"),
        model_path=os.path.join(
            "trained_model",
            "0817_8786_concat_num",
            "simqasl",
            "2020-12-31-03-annotator-model-dev@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-8_E-25_LR-5e-5_SD-1",
        ),
    )
    da_a.augment(
        input_data_path=os.path.join("data", "final", in_v, target_dataset + ".json"),
        p_times=1.3,
        output_data_path=os.path.join("data", "final", out_v, target_dataset + ".json"),
    )

    # --- Squad2Df (to easily be checked by human) ---
    in_v = "V1.1"
    out_v = "V1.1"
    dt_v0a = DataTransformer(
        version=in_v, time=datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    )
    dt_v0a.squad2df(
        input_data_path=os.path.join("data", "final", in_v, target_dataset + ".json"),
        output_data_path=os.path.join(
            "data", "final", out_v, "1wait4bechecked", target_dataset + "_wait4bechecked.xlsx"
        ),
    )

    # --- Df2Squad ---
    in_v = "V1.1"
    out_v = "V1.1c"
    dt_v1a = DataTransformer(
        version=out_v, time=datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    )
    dt_v1a.df2squad(
        input_data_path=os.path.join(
            "data", "final", in_v, "3checked8programming", "3train_checked_all.xlsx"
        ),
        output_data_path=os.path.join(
            "data", "final", out_v, target_dataset + ".json"
        ),
    )