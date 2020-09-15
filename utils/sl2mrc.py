# encoding=utf-8
# Author: Allen.Chiang
# Description: transform sequence labeling to machine reading comprehension

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
    global answers
    global temp_ans
    global idx
    passage = ""
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

with open (os.path.join("configs", "mrc_query.json"), "r") as f: 
    mrc_query = json.load(f)

for dataset in ["train.txt", "dev.txt", "test.txt"]:
    mrc_data = dict()
    mrc_data["version"] = version
    mrc_data["query"] = mrc_query
    mrc_data["data"] = list()

    init_parent()
    init_child()

    pid = 0
    with open (os.path.join("data", "sl", dataset), "r") as fin:
        for i, line in tqdm(enumerate(fin)):
            line = line.rstrip()
            if line == "":
                mrc_data["data"].append(
                    {
                        "pid": pid,
                        "passage": passage,
                        "answers": temp_ans,
                    }
                )
                pid += 1
                init_parent()
            else:
                char, label = line.split()
                flag, cate = label.split("-") if label != "O" else (None, label)
                if label == "O" and text != "":
                    end_pos = idx-1
                    temp_ans.append(
                        {
                            "text": text,
                            "label": temp_cate,
                            "start_pos": start_pos,
                            "end_pos": end_pos
                        }
                    )
                    init_child()
                
                elif label != "O":
                    if temp_cate == "":
                        start_pos = idx

                    elif temp_cate != "" and cate != temp_cate:
                        end_pos = idx-1
                        temp_ans.append(
                            {
                                "text": text,
                                "label": temp_cate,
                                "start_pos": start_pos,
                                "end_pos": end_pos
                            }
                        )
                        init_child()
                        start_pos = idx

                    text += char
                    temp_cate = cate

                passage += char
                idx += 1
        
    with open (os.path.join("data", "mrc", dataset), "w") as fout:
        mrc_data_json = json.dumps(mrc_data, indent=4, ensure_ascii=False)
        fout.write(mrc_data_json)
