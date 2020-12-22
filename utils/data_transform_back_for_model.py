# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform qasl/simqasl data to txt file

import os
import json
from tqdm import tqdm
import pandas as pd


ref_data_path = os.path.join(
    "data", "simqasl", "0817_8786_concat_num", f"loop_1", "test.txt"
)
in_data_path = os.path.join(
    "data",
    "simqasl",
    "0817_8786_concat_num",
    f"loop_1_checked",
    "test_20201211_381.xlsx",
)
out_data_path = os.path.join(
    "data",
    "simqasl",
    "0817_8786_concat_num",
    f"loop_1_checked",
    "test_20201211_381.txt",
)


with open(ref_data_path, "r", encoding="utf-8") as f:
    mrc_data = json.load(f)
version = mrc_data["version"]
query = mrc_data["query"]
data = mrc_data["data"]

loop_1_checked = pd.read_excel(in_data_path)
cols = loop_1_checked.columns.tolist()[1:]  ## drop passage

for d in data:
    pid = d["pid"]
    passage = d["passage"]
    passage_tokens = d["passage_tokens"]
    ans_origin = d["answers"]

    ## create
    d["checked_origin_ans"] = list()  ## None or [o] or [ok]
    d["checked_model_pred"] = list()  ## [m]
    d["checked_karen"] = list()  ## [k]

    for col in cols:
        manual_check_ans_text = loop_1_checked.loc[pid, col]
        if pd.isna(manual_check_ans_text):
            pass
        else:
            manual_check_ans_text = manual_check_ans_text.strip(", ")  ## remove space and comma both at the beginning and end of string
            split_ans_list = manual_check_ans_text.split(", ")
            split_ans_list = list(filter(None, split_ans_list))  ## remove empty strings
            for each_ans in split_ans_list:
                each_ans = (each_ans.strip())  ## remove space both at the beginning and end of string
                print(f"{pid}, {col}, {each_ans}")

                a = each_ans.find("[")
                b = each_ans.find("]")
                ans_text = each_ans[:a]
                loc_list = each_ans[a + 1 : b].split("-")
                start_pos = int(loc_list[0])
                end_pos = int(loc_list[1])
                ans_double_checked = "".join(passage_tokens[start_pos : end_pos + 1])

                if ans_text != ans_double_checked:
                    raise ValueError(
                        f"{pid}, {col}, {ans_text}, {start_pos}-{end_pos}, {ans_double_checked}"
                    )

                if "[m]" in each_ans:
                    d["checked_model_pred"].append(
                        {
                            "text": ans_text,
                            "label": col,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )

                elif "[k]" in each_ans:
                    d["checked_karen"].append(
                        {
                            "text": ans_text,
                            "label": col,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )

                else:  ## None, [o], [ok]
                    d["checked_origin_ans"].append(
                        {
                            "text": ans_text,
                            "label": col,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )


with open(out_data_path, "w", encoding="utf-8") as fout:
    out = json.dumps(mrc_data, indent=4, ensure_ascii=False)
    fout.write(out)
