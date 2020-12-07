# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform qasl/simqasl data to dataframe and then check correctness

import os
import json
import pandas as pd

loop = 1
data_dir = os.path.join("data", "simqasl", "0817_8786_concat_num", f"loop_{loop}")
data_path = os.path.join(data_dir, "test.txt")
out_path = os.path.join(data_dir, "test.xlsx")

with open(data_path, "r", encoding="utf-8") as f:
    mrc_data = json.load(f)
version = mrc_data["version"]
query = mrc_data["query"]
data = mrc_data["data"]

cols = ["passage"] + list(query.keys()) + ["DIN"]
df = pd.DataFrame(columns=cols)

for d in data:
    pid = d["pid"]
    df.loc[pid, "passage"] = d["passage"]
    
    for a in d["answers"]:
        text = a["text"]
        label = a["label"]
        start_pos = a["start_pos"]
        end_pos = a["end_pos"]
        if pd.isna(df.loc[pid, label]):
            df.loc[pid, label] = f"{text}[{start_pos}-{end_pos}]"
        else:
            df.loc[pid, label] += ", " + f"{text}[{start_pos}-{end_pos}]"
    
    for a in d["model_pred_top2"]:
        text = a["text"]
        label = a["label"]
        start_pos = a["start_pos"]
        end_pos = a["end_pos"]
        if pd.isna(df.loc[pid, label]):
            df.loc[pid, label] = f"{text}[{start_pos}-{end_pos}][m]"
        else:
            df.loc[pid, label] += ", " + f"{text}[{start_pos}-{end_pos}][m]"

df.to_excel(out_path, index=False)