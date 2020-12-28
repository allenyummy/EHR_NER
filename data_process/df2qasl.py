# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform qasl/simqasl data to txt file

import os
import json
import datetime
from tqdm import tqdm
import pandas as pd


ref_data_path = os.path.join(
    "data", "final", "dev.json"
)
in_data_path = os.path.join(
    "data",
    "simqasl",
    "0817_8786_concat_num",
    "loop_1_checked",
    "dev_checked.xlsx",
)
out_data_path = os.path.join(
    "data",
    "final",
    "loop_1_checked",
    "dev_checked.json",
)


with open(ref_data_path, "r", encoding="utf-8") as f:
    in_data = json.load(f)
in_data["version"] = in_data["version"] + "_checked"
in_data["time"] = datetime.datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
data = in_data["data"]

loop_1_checked = pd.read_excel(in_data_path)
cols = loop_1_checked.columns.tolist()[1:]  ## drop passage

for i, d in enumerate(data):

    pid = d["pid"]
    passage = d["passage"]
    passage_tokens = d["passage_tokens"]

    ## create
    d["checked_origin_ans"] = list()  ## None or [o] or [ok]
    d["checked_model_pred"] = list()  ## [m]
    d["checked_karen"] = list()  ## [k]

    ## create answers
    d["concat_answers"] = list()
    d["flat_ne_answers"] = list()
    d["nested_ne_answers"] = list()

    for col in cols:
        manual_check_ans_text = loop_1_checked.loc[pid, col]
        if pd.isna(manual_check_ans_text):
            pass
        else:
            # ---
            manual_check_ans_text = manual_check_ans_text.strip(
                ", "
            )  ## remove space and comma both at the beginning and end of string
            split_ans_list = manual_check_ans_text.split(", ")
            split_ans_list = list(filter(None, split_ans_list))  ## remove empty strings
            for each_ans in split_ans_list:
                each_ans = (
                    each_ans.strip()
                )  ## remove space both at the beginning and end of string

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
                            "type": col,
                            "text": ans_text,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )

                elif "[k]" in each_ans:
                    d["checked_karen"].append(
                        {
                            "type": col,
                            "text": ans_text,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )

                else:  ## None, [o], [ok]
                    d["checked_origin_ans"].append(
                        {
                            "type": col,
                            "text": ans_text,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )

    # --- concat all checked answers ---
    concat_ans = d["checked_karen"] + d["checked_origin_ans"]

    # --- generate flat ne answers ---
    seen_ans = set()
    before_flat_ne_answers = list()
    flat_ne_answers = list()
    priority = {
        "OPC": 0.01,
        "EMC": 0.01,
        "CTC": 0.02,
        "RTC": 0.02,
        "ADD": 0,
        "OPD": 1,
        "OPDS": 1.5,
        "OPDE": 1.5,
        "CTD": 1.6,
        "CTDS": 1.7,
        "RTD": 1.6,
        "RTDS": 1.7,
        "EMD": 2,
        "EMDS": 2.5,
        "EMDE": 2.5,
        "IND": 2.6,
        "ICD": 2.6,
        "IBD": 2.6,
        "SGD": 3,
        "SGDS": 3.5,
        "DCD": 5,
        "OCD": 5.1,
        "OND": 5.1,
        "OBD": 5.1,
        "CTDE": 5.1,
        "RTDE": 5.1,
        "SGDE": 5.1,
    }

    for a in concat_ans:
        b = tuple(a.items())
        if b not in seen_ans:
            before_flat_ne_answers.append(a)
            seen_ans.add(b)

    for a in sorted(
        before_flat_ne_answers, key=lambda x: (x["start_pos"], x["end_pos"], x["type"])
    ):
        if not flat_ne_answers:
            flat_ne_answers.append(a)
        else:
            temp_ans = flat_ne_answers.pop()
            if (
                a["start_pos"] == temp_ans["start_pos"]
                and a["end_pos"] == temp_ans["end_pos"]
            ):
                label_a = a["type"]
                label_temp = temp_ans["type"]
                passage_tokens_string = " ".join(passage_tokens)

                print(f"{pid}, {passage_tokens_string}")
                print(
                    f"{a['start_pos']}-{a['end_pos']}, {a['text']}, {label_a} <-> {label_temp}",
                    end=", ",
                )
                if priority[label_a] > priority[label_temp]:
                    flat_ne_answers.append(a)
                    print(f"{label_a}")
                else:
                    flat_ne_answers.append(temp_ans)
                    print(f"{label_temp}")
                print("---")
            else:
                flat_ne_answers.append(temp_ans)
                flat_ne_answers.append(a)

    flat_ne_answers = sorted(
        flat_ne_answers, key=lambda x: (x["start_pos"], x["end_pos"], x["type"])
    )
    d["flat_ne_answers"] = flat_ne_answers

    # --- generate nested ne answers from concat answers ---
    concat_ans = concat_ans + d["checked_model_pred"]
    seen_ans = set()
    nested_ne_answers = list()
    for a in concat_ans:
        b = tuple(a.items())
        if b not in seen_ans:
            nested_ne_answers.append(a)
            seen_ans.add(b)
    nested_ne_answers = sorted(
        nested_ne_answers, key=lambda x: (x["start_pos"], x["end_pos"], x["type"])
    )
    d["nested_ne_answers"] = nested_ne_answers

    diff_answers = nested_ne_answers.copy()
    for i in flat_ne_answers:
        diff_answers.remove(i)
    d["diff_answers"] = diff_answers

    del d["concat_answers"]
    del d["checked_origin_ans"]
    del d["checked_model_pred"]
    del d["checked_karen"]


with open(out_data_path, "w", encoding="utf-8") as fout:
    out = json.dumps(in_data, indent=4, ensure_ascii=False)
    fout.write(out)
