# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform sequence labeling to QA-based sequence labeling

import os
import json
import logging
import datetime
from tqdm import tqdm, trange
from collections import defaultdict

logger = logging.getLogger(__name__)

time = datetime.datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
version = "v1.0_" + time


def init_parent():
    global passage
    global passage_tokens
    global answers
    global temp_ans
    global idx
    passage = ""
    passage_tokens = list()
    answers = list()
    temp_ans = list()
    idx = 0


def init_child():
    global text
    global temp_cate
    global start_pos
    global end_pos
    text = ""
    temp_cate = ""
    start_pos = 0
    end_pos = 0


with open(os.path.join("configs", "qasl_simple_query.json"), "r", encoding="utf-8") as f:
    mrc_query = json.load(f)

for dataset in ["train.txt", "dev.txt", "test.txt"]:
    mrc_data = dict()
    mrc_data["version"] = version
    mrc_data["query"] = mrc_query
    mrc_data["data"] = list()

    init_parent()
    init_child()

    pid = 0
    with open(os.path.join("data", "sl", "0817_8786_concat_num", dataset), "r", encoding="utf-8") as fin:
        for i, line in tqdm(enumerate(fin)):
            line = line.rstrip()
            if line == "":

                if text != "":
                    end_pos = idx - 1
                    temp_ans.append(
                        {
                            "text": text,
                            "label": temp_cate,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )
                    init_child()

                mrc_data["data"].append(
                    {
                        "pid": pid,
                        "passage": passage,
                        "passage_tokens": passage_tokens,
                        "answers": temp_ans,
                    }
                )
                pid += 1
                init_parent()
            else:
                char, label = line.split()
                flag, cate = label.split("-") if label != "O" else (None, label)
                if label == "O" and text != "":
                    end_pos = idx - 1
                    temp_ans.append(
                        {
                            "text": text,
                            "label": temp_cate,
                            "start_pos": start_pos,
                            "end_pos": end_pos,
                        }
                    )
                    init_child()

                elif label != "O":
                    if temp_cate == "":
                        start_pos = idx

                    elif temp_cate != "" and cate != temp_cate:
                        end_pos = idx - 1
                        temp_ans.append(
                            {
                                "text": text,
                                "label": temp_cate,
                                "start_pos": start_pos,
                                "end_pos": end_pos,
                            }
                        )
                        init_child()
                        start_pos = idx

                    text += char
                    temp_cate = cate

                passage += char
                passage_tokens.append(char)
                idx += 1

        if passage != "" and temp_ans:
            mrc_data["data"].append(
                {
                    "pid": pid,
                    "passage": passage,
                    "passage_tokens": passage_tokens,
                    "answers": temp_ans,
                }
            )

    with open(os.path.join("data", "simqasl", "0817_8786_concat_num", dataset), "w") as fout:
        mrc_data_json = json.dumps(mrc_data, indent=4, ensure_ascii=False)
        fout.write(mrc_data_json)
