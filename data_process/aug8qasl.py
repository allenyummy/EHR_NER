# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: data augmentaion for qasl and simqasl

import logging
import os
import re
import json
from tqdm import tqdm
from utils.re_datetime import Pattern
from api.bert_qasl_predictor import BertQASLPredictor
from src.scheme import IOB2
from src.entity import EntityFromList, EntityElements

logger = logging.getLogger(__name__)


def predict(model, query, passage, threshold):
    results_top1 = list()
    results_top2 = list()
    for t, q in query.items():
        res = model.predict(t, q, passage, top_k=2)
        token, l1, p1, l2, p2 = zip(*res)
        seq_top1 = [(t, l) for t, l in zip(token, l1)]
        results_top1.extend(EntityFromList(seq=seq_top1, scheme=IOB2).entities)
        seq_top2 = [(t, l if p > threshold else "O") for t, l, p in zip(token, l2, p2)]
        results_top2.extend(EntityFromList(seq=seq_top2, scheme=IOB2).entities)
    results_top2_prune = pat_prune(results_top2)
    return results_top1, results_top2, results_top2_prune


def pat_prune(results):
    results_prune = list()
    for res in results:
        if any(res.type.endswith(s) for s in ["D", "S", "E"]):
            check_pat1 = re.search(Pattern.pat1.value, res.text)
            if check_pat1 and res.text == check_pat1.group():
                results_prune.append(res)
            check_pat2 = re.search(Pattern.pat2.value, res.text)
            if check_pat2 and res.text == check_pat2.group():
                results_prune.append(res)
    return results_prune


def logging(passage, ans, results_top1, results_top2, results_top2_prune):
    print(f"passage: {passage}")
    print("==== Ans ===")
    for a in ans:
        print (EntityElements(pid, a["label"], a["start_pos"], a["end_pos"], a["text"]))
    print("==== Top 1 prediction ====")
    for ent in sorted(results_top1, key=lambda x: x.start_pos):
        print(ent)
    # print(f"==== Top 2 prediction ====")
    # for ent in sorted(results_top2, key=lambda x: x.start_pos):
    #     print(ent)
    print(f"==== Top 2 prediction after pruning ====")
    for ent in sorted(results_top2_prune, key=lambda x: x.start_pos):
        print(ent)


if __name__ == "__main__":
    loop = 1
    # model_dir = "trained_model/0817_8786_concat_num/bert_qasl/2020-11-11-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-4_E-5_LR-5e-5_SD-1/"
    # threshold = 0.00001
    # data_dir = os.path.join("data", "qasl", "0817_8786_concat_num")
    # data_path = os.path.join(data_dir, "test.txt")
    # out_path = os.path.join(data_dir, f"loop_{loop}", "test.txt")

    model_dir = "trained_model/0817_8786_concat_num/bert_simqasl/2020-11-20-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1/"
    threshold = 0.0000008
    data_dir = os.path.join("data", "simqasl", "0817_8786_concat_num")
    data_path = os.path.join(data_dir, "test.txt")
    out_path = os.path.join(data_dir, f"loop_{loop}", "test.txt")

    model = BertQASLPredictor(model_dir)
    
    with open(data_path, "r", encoding="utf-8") as f:
        mrc_data = json.load(f)
    version = mrc_data["version"]
    query = mrc_data["query"]
    data = mrc_data["data"]

    for d in tqdm(data):
        pid = d["pid"]
        passage = d["passage"]
        passage_tokens = d["passage_tokens"]
        ans = d["answers"]
        results_top1, results_top2, results_top2_prune = predict(model, query, " ".join(passage_tokens), threshold)
        
        model_pred_top2 = list()
        for add_ans in results_top2_prune:
            model_pred_top2.append(
                {
                    "text": add_ans.text,
                    "label": add_ans.type,
                    "start_pos": add_ans.start_pos,
                    "end_pos": add_ans.end_pos,
                }
            )
        d["model_pred_top2"] = model_pred_top2
    
    with open(out_path, "w", encoding="utf-8") as fout:
        out = json.dumps(mrc_data, indent=4, ensure_ascii=False)
        fout.write(out)
        

