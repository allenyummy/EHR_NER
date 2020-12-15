# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for api/bert_qasl_predictor.py

import logging
import pytest
import json
import re
from utils.re_datetime import Pattern
from api.bert_qasl_predictor import BertQASLPredictor
from src.scheme import IOB2
from src.entity import EntityFromList

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def model():
    model_dir = "trained_model/0817_8786_concat_num/qasl/2020-11-11-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-4_E-5_LR-5e-5_SD-1/"
    BertQASL = BertQASLPredictor(model_dir)
    return BertQASL


@pytest.fixture(scope="function")
def query():
    qasl_dir = "configs/qasl_query.json"
    with open(qasl_dir, "r", encoding="utf-8") as f:
        qasl_query = json.load(f)
    return qasl_query


def predict(model, query, passage, threshold=0.00001):
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


def logging(passage, results_top1, results_top2, results_top2_prune):
    logger.info(f"passage: {passage}")
    logger.info("==== Top 1 prediction ====")
    for ent in sorted(results_top1, key=lambda x: x.start_pos):
        logger.info(ent)
    # logger.info(f"==== Top 2 prediction ====")
    # for ent in sorted(results_top2, key=lambda x: x.start_pos):
    #     logger.info(ent)
    logger.info(f"==== Top 2 prediction after pruning ====")
    for ent in sorted(results_top2_prune, key=lambda x: x.start_pos):
        logger.info(ent)


def testcase_one(model, query, testcase1):
    passage = testcase1["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_two(model, query, testcase2):
    passage = testcase2["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_three(model, query, testcase3):
    passage = testcase3["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_four(model, query, testcase4):
    passage = testcase4["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_five(model, query, testcase5):
    passage = testcase5["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_six(model, query, testcase6):
    passage = testcase6["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_seven(model, query, testcase7):
    passage = testcase7["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_eight(model, query, testcase8):
    passage = testcase8["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_nine(model, query, testcase9):
    passage = testcase9["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_ten(model, query, testcase10):
    passage = testcase10["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_eleven(model, query, testcase11):
    passage = testcase11["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_twelve(model, query, testcase12):
    passage = testcase12["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_thirteen(model, query, testcase13):
    passage = testcase13["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_fourteen(model, query, testcase14):
    passage = testcase14["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_fifteen(model, query, testcase15):
    passage = testcase15["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_sixteen(model, query, testcase16):
    passage = testcase16["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_seventeen(model, query, testcase17):
    passage = testcase17["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_eighteen(model, query, testcase18):
    passage = testcase18["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)


def testcase_nineteen(model, query, testcase19):
    passage = testcase19["passage"]
    results_top1, results_top2, results_top2_prune = predict(model, query, passage)
    logging(passage, results_top1, results_top2, results_top2_prune)