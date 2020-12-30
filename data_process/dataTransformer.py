# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform dataset from traditional tagging style to SQuAD style

import re
import os
import json
import logging
import datetime
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm
from src.entity import EntityFromNestedList
from src.scheme import IOB2

logger = logging.getLogger(__name__)


class DataTransformer:
    def __init__(self, version: str, time: str):
        self.version = version
        self.time = time

    def tag2squad(self, input_data_path: str, output_data_path: str):
        # --- Read data
        with open(input_data_path, "r", encoding="utf-8") as fin:
            in_data = fin.read().splitlines()
            # --- assure last element of data is an empty string because we use it as the end of a passage
            if in_data[-1] != "":
                in_data = in_data + [""]

        # --- Transfrom list data into nested list which has tuple elements ("民", "B-ADD")
        seqs: List[List[Tuple[str, str]]] = list()
        seq: List[[Tuple[str, str]]] = list()
        for d in in_data:
            if d == "":  # we use it as the end of a passage
                seqs.append(seq)
                seq = list()
            else:
                char = d.split()[0]
                label = d.split()[1]
                seq.append((char, label))

        # --- Get start and end position of each passage from nested list by seqhelper package
        entities = EntityFromNestedList(seqs=seqs, scheme=IOB2).entities

        # --- Output data
        out_data = dict()
        out_data["version"] = self.version
        out_data["time"] = self.time
        out_data["data"] = list()
        for pid, seq in enumerate(seqs):
            passage_tokens, label = zip(*seq)
            passage_tokens = list(passage_tokens)
            passage = "".join(passage_tokens)
            out_data["data"].append(
                {
                    "pid": pid,
                    "passage": passage,
                    "passage_tokens": passage_tokens,
                    "flat_ne_answers": [
                        {
                            "type": ent.type,
                            "text": ent.text,
                            "start_pos": ent.start_pos,
                            "end_pos": ent.end_pos,
                        }
                        for ent in entities[pid]
                    ],
                }
            )

        # --- Write to json file
        with open(output_data_path, "w") as fout:
            out_data_json = json.dumps(out_data, indent=4, ensure_ascii=False)
            fout.write(out_data_json)
        return out_data

    def squad2tag(self, input_data_path: str, output_data_path: str):
        raise NotImplementedError

    def squad2df(self, input_data_path: str, output_data_path: str):
        # --- Read data
        with open(input_data_path, "r", encoding="utf-8") as fin:
            in_data = json.load(fin)
        data = in_data["data"]

        # --- Transform squad data into Dataframe with four columns: `pid`, `passage`, `passage_tokens`, `flat_ne_answers`, `nested_ne_answers`
        df = pd.DataFrame(
            columns=[
                "pid",
                "passage",
                "passage_tokens",
                "flat_ne_answers",
                "nested_ne_answers",
            ]
        )
        for i, d in enumerate(tqdm(data)):
            pid = d["pid"]
            passage = d["passage"]
            passage_tokens = d["passage_tokens"]
            flat_ne_answers = d["flat_ne_answers"]
            nested_ne_answers = d["nested_ne_answers"]

            flat_ne_ans_lst = list()
            for ans in flat_ne_answers:
                type = ans["type"]
                text = ans["text"]
                start_pos = ans["start_pos"]
                end_pos = ans["end_pos"]
                o = f"[{type}][{text}][{start_pos}-{end_pos}]"
                flat_ne_ans_lst.append(o)

            nested_ne_ans_lst = list()
            for ans in nested_ne_answers:
                type = ans["type"]
                text = ans["text"]
                start_pos = ans["start_pos"]
                end_pos = ans["end_pos"]
                o = f"[{type}][{text}][{start_pos}-{end_pos}]"
                nested_ne_ans_lst.append(o)

            # --- Put data into right cell of dataframe
            df.loc[pid, "pid"] = pid
            df.loc[pid, "passage"] = passage
            df.loc[pid, "passage_tokens"] = " ".join(passage_tokens)
            df.loc[pid, "flat_ne_answers"] = ",\n".join(flat_ne_ans_lst)
            df.loc[pid, "nested_ne_answers"] = ",\n".join(nested_ne_ans_lst)

            # --- Save it every ten data in the middle of processing
            if i % 10 == 0:
                df.to_excel(output_data_path, index=False)
        df.to_excel(output_data_path, index=False)
        return df

    def df2squad(self, input_data_path: str, output_data_path: str):
        # --- Read data
        data = pd.read_excel(input_data_path)

        # --- Transform dataframe into squad
        out_data = dict()
        out_data["version"] = self.version
        out_data["time"] = self.time
        out_data["data"] = list()

        for i in tqdm(range(len(data))):
            pid = data.loc[i, "pid"]
            passage = data.loc[i, "passage"]
            passage_tokens = data.loc[i, "passage_tokens"].split()
            flat_ne_answers = data.loc[i, "flat_ne_answers"].split(",\n")
            nested_ne_answers = data.loc[i, "nested_ne_answers"].split(",\n")

            flat_ne_answers_lst = list()
            for ans in flat_ne_answers:
                res = re.search(r"\[(.+)\]\[(.+)\]\[(\d+)-(\d+)\]", ans).groups()
                flat_ne_answers_lst.append(
                    {
                        "type": res[0],
                        "text": res[1],
                        "start_pos": int(res[2]),
                        "end_pos": int(res[3]),
                    }
                )

            nested_ne_answers_lst = list()
            for ans in nested_ne_answers:
                res = re.search(r"\[(.+)\]\[(.+)\]\[(\d+)-(\d+)\]", ans).groups()
                nested_ne_answers_lst.append(
                    {
                        "type": res[0],
                        "text": res[1],
                        "start_pos": int(res[2]),
                        "end_pos": int(res[3]),
                    }
                )

            out_data["data"].append(
                {
                    "pid": int(pid),
                    "passage": passage,
                    "passage_tokens": passage_tokens,
                    "flat_ne_answers": flat_ne_answers_lst,
                    "nested_ne_answers": nested_ne_answers_lst,
                }
            )
        with open(output_data_path, "w") as fout:
            out_data_json = json.dumps(out_data, indent=4, ensure_ascii=False)
            fout.write(out_data_json)
        return out_data


if __name__ == "__main__":

    version = "v0.1c"
    time = datetime.datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    dt = DataTransformer(version, time)

    # # --- Demo for tag2squad function
    # for in_dataset, out_dataset in zip(["train.txt", "dev.txt", "test.txt"], ["train.json", "dev.json", "test.json"]):
    #     logger.warning(f"Processing {in_dataset} ...")
    #     input_data_path = os.path.join("data", "sl", "0817_8786_concat_num", in_dataset)
    #     output_data_path = os.path.join("data", "final", out_dataset)
    #     out_data = dt.tag2squad(input_data_path, output_data_path)
    #     logger.warning(f" ===> {out_dataset}, Done !")

    # # --- Demo for squad2df function
    # input_data_path = os.path.join("data", "final", "V0.1", "dev.json")
    # output_data_path = os.path.join("data", "final", "V0.1", "1. df_wait4bechecked", "dev_wait4bechecked.xlsx")
    # out_data = dt.squad2df(input_data_path, output_data_path)

    # --- Demo for df2squad function
    input_data_path = os.path.join("data", "final", "V0.1c", "1. df_checked", "dev_checked.xlsx")
    output_data_path = os.path.join("data", "final", "V0.1c", "dev.json")
    out_data = dt.df2squad(input_data_path, output_data_path)
