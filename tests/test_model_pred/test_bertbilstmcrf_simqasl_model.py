# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for api/bert_qasl_predictor.py

import logging
import pytest
import json
import re
from api.bert_qasl_predictor import BertQASLPredictor
from src.scheme import IOB2
from src.entity import EntityFromList

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def model():
    model_dir = "trained_model/0817_8786_concat_num/simqasl/2020-12-17-07@hfl@chinese-bert-wwm@wBiLSTMCRF-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1"
    model = BertQASLPredictor(model_dir, with_bilstmcrf=True)
    return model


@pytest.fixture(scope="function")
def query():
    simqasl_dir = "configs/simqasl_query.json"
    with open(simqasl_dir, "r", encoding="utf-8") as f:
        simqasl_query = json.load(f)
    return simqasl_query


def predict(model, query, passage):
    results = list()
    for t, q in query.items():
        res = model.predict(t, q, passage)
        # for r in res:
        #     logger.info(r)
        token, label = zip(*res)
        seq = [(t, l) for t, l in zip(token, label)]
        results.extend(EntityFromList(seq=seq, scheme=IOB2).entities)
    return results


def logging(passage, results):
    logger.info(f"passage: {passage}")
    logger.info("==== Top 1 prediction ====")
    for ent in sorted(results, key=lambda x: x.start_pos):
        logger.info(ent)


def testcase_one(model, query, testcase1):
    passage = testcase1["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_two(model, query, testcase2):
    passage = testcase2["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_three(model, query, testcase3):
    passage = testcase3["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_four(model, query, testcase4):
    passage = testcase4["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_five(model, query, testcase5):
    passage = testcase5["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_six(model, query, testcase6):
    passage = testcase6["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_seven(model, query, testcase7):
    passage = testcase7["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_eight(model, query, testcase8):
    passage = testcase8["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_nine(model, query, testcase9):
    passage = testcase9["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_ten(model, query, testcase10):
    passage = testcase10["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_eleven(model, query, testcase11):
    passage = testcase11["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_twelve(model, query, testcase12):
    passage = testcase12["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_thirteen(model, query, testcase13):
    passage = testcase13["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_fourteen(model, query, testcase14):
    passage = testcase14["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_fifteen(model, query, testcase15):
    passage = testcase15["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_sixteen(model, query, testcase16):
    passage = testcase16["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_seventeen(model, query, testcase17):
    passage = testcase17["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_eighteen(model, query, testcase18):
    passage = testcase18["passage"]
    results = predict(model, query, passage)
    logging(passage, results)


def testcase_nineteen(model, query, testcase19):
    passage = testcase19["passage"]
    results = predict(model, query, passage)
    logging(passage, results)