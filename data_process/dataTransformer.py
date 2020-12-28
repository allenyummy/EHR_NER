# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: transform dataset from traditional tagging style to SQuAD style

import os
import json
import logging
import datetime
from typing import List, Tuple
from src.entity import EntityFromNestedList
from src.scheme import IOB2

logger = logging.getLogger(__name__)


class DataTransformer:
    def __init__(self, version, time):
        self.version = version
        self.time = time

    def tag2squad(self, input_data_path, output_data_path):
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

    def squad2tag(self, input_data_path, output_data_path):
        raise NotImplementedError


if __name__ == "__main__":
    version = "v2.0"
    time = datetime.datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    dt = DataTransformer(version, time)

    for in_dataset, out_dataset in zip(["train.txt", "dev.txt", "test.txt"], ["train.json", "dev.json", "test.json"]):
        logger.warning(f"Processing {in_dataset} ...")
        input_data_path = os.path.join("data", "sl", "0817_8786_concat_num", in_dataset)
        output_data_path = os.path.join("data", "final", out_dataset)
        out_data = dt.tag2squad(input_data_path, output_data_path)
        logger.warning(f" ===> {out_dataset}, Done !")