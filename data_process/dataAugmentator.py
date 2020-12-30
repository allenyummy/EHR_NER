# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: data augmentaion by BertQASLModel

import logging
import os
import json
import datetime
import numpy as np
from tqdm import tqdm
from api.bert_qasl_predictor import BertQASLPredictor
from src.scheme import IOB2
from src.entity import EntityFromList, EntityElements

logger = logging.getLogger(__name__)


class DataAugmentator:
    def __init__(self, version, time, query_path, model_path):
        self.version = version
        self.time = time
        with open(query_path, "r", encoding="utf-8") as fq:
            self.query = json.load(fq)
        self.model_path = model_path
        self.model = BertQASLPredictor(model_path)
        self.label_map = {i: label for i, label in enumerate(self.query.keys())}

    def augment(self, input_data_path, p_times, output_data_path):
        # --- Read data
        with open(input_data_path, "r", encoding="utf-8") as fin:
            in_data = json.load(fin)
        data = in_data["data"]
        in_data["version"] = self.version
        in_data["time"] = self.time
        in_data["aug8model"] = f"{self.model_path}"

        # --- Augment
        for i, d in enumerate(tqdm(data)):
            pid = d["pid"]
            passage = d["passage"]
            passage_tokens = d["passage_tokens"]
            input_passage = " ".join(passage_tokens)
            flat_ne_answers = d["flat_ne_answers"]
            res_top1, res_top2_pruned = self._refine_predict(input_passage, p_times)

            # --- transform to dict
            trans = list()
            concat_pred = list()
            if res_top1:
                concat_pred.extend(res_top1)
            if res_top2_pruned:
                concat_pred.extend(res_top2_pruned)
            for r in concat_pred:
                rd = dict(r._asdict())
                del rd["pid"]
                trans.append(rd)

            # --- produce nested_ne_answers
            seen_ans = set()
            nested_ne_answers = list()
            for a in trans:
                b = tuple(a.items())
                if b not in seen_ans:
                    nested_ne_answers.append(a)
                    seen_ans.add(b)
            nested_ne_answers = sorted(
                nested_ne_answers,
                key=lambda x: (x["start_pos"], x["end_pos"], x["type"]),
            )
            d["nested_ne_answers"] = nested_ne_answers

            # --- logging
            if i < 5:
                logger.warning("")
                logger.warning("***********************************************************")
                logger.warning(f"************************* PID {i} ***************************")
                logger.warning("***********************************************************")
                logger.warning(f"passage: {passage}")
                logger.warning("===== FLAT NE ANSWER =====")
                for a in flat_ne_answers:
                    logger.warning(
                        EntityElements(
                            pid, a["type"], a["start_pos"], a["end_pos"], a["text"]
                        )
                    )
                logger.warning("===== TOP1 PRED =====")
                for a in sorted(res_top1, key=lambda x: x.start_pos):
                    logger.warning(a)
                logger.warning("===== TOP2 PRED =====")
                for a in sorted(res_top2_pruned, key=lambda x: x.start_pos):
                    logger.warning(a)

        # --- Write to json file
        with open(output_data_path, "w") as fout:
            out_data_json = json.dumps(in_data, indent=4, ensure_ascii=False)
            fout.write(out_data_json)

        return in_data

    def _refine_predict(self, passage, p_times):
        res_top1 = list()
        res_top2 = list()
        for tag, q in self.query.items():
            res = self.model.predict(tag, q, passage, top_k=2)
            tokens, l1, p1, l2, p2 = zip(*res)
            seq_top1 = [(token, l) for token, l in zip(tokens, l1)]
            seq_top2 = [(l, p) for l, p in zip(l2, p2)]
            res_top1.extend(EntityFromList(seq=seq_top1, scheme=IOB2).entities)
            res_top2.append(seq_top2)
        res_top2_pruned = self._get_back_from_top2(res_top1, res_top2, p_times)
        return res_top1, res_top2_pruned

    def _get_back_from_top2(self, res_top1, res_top2, p_times):
        # --- Get date position from res_top1
        date_position = list()
        text_list = list()
        for a in sorted(res_top1, key=lambda x: x.start_pos):
            if any(a.type.endswith(s) for s in ["D", "S", "E"]):
                date_position.append((a.start_pos, a.end_pos))
                text_list.append(a.text)

        # --- Get back prediction from res_top2
        res_top2_pruned = list()
        for j, (start_pos, end_pos) in enumerate(date_position):
            for i, top2 in enumerate(res_top2):
                type = self.label_map[i]
                mean = np.mean([t[1] for t in top2])
                p_threshold = mean * p_times
                if top2[start_pos][0].startswith("B") and all(
                    t[0].startswith("I") for t in top2[start_pos + 1 : end_pos + 1]
                ):
                    end_mean = np.mean(
                        [t[1] for t in top2[start_pos + 1 : end_pos + 1]]
                    )
                    if top2[start_pos][1] > p_threshold and end_mean > p_threshold:
                        ent_mean = np.mean(
                            [t[1] for t in top2[start_pos : end_pos + 1]]
                        )
                        if ent_mean > p_threshold:
                            text = text_list[j]
                            res_top2_pruned.append(
                                EntityElements(
                                    ent_mean / mean, type, start_pos, end_pos, text
                                )
                            )
        return res_top2_pruned


if __name__ == "__main__":
    version = "v0.1"
    time = datetime.datetime.today().strftime("%Y/%m/%d_%H:%M:%S")
    query_path = os.path.join("data", "final", "query", "simqasl_query.json")
    model_path = "trained_model/0817_8786_concat_num/simqasl/2020-12-10-07@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1/"
    p_times = 1.3
    input_data_path = "data/final/V0.0/dev.json"
    output_data_path = "data/final/V0.1/dev.json"
    da = DataAugmentator(version, time, query_path, model_path)
    out_data = da.augment(input_data_path, p_times, output_data_path)
