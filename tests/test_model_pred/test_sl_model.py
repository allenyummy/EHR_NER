# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for api/bert_sl_predictor.py

import logging
import pytest
from api.bert_sl_predictor import BertSLPredictor
from src.scheme import IOB2
from src.entity import EntityFromList

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def model():
    model_dir = "trained_model/0817_8786_concat_num/bert_sl/2020-09-02-00@hfl@chinese-bert-wwm@CE_S-512_B-4_E-100_LR-5e-5_SD-1/"
    BertSL = BertSLPredictor(model_dir)
    return BertSL


def predict(model, passage):
    res = model.predict(passage)
    token, label, prob = zip(*res)
    seq = [(t, l) for t, l in zip(token, label)]
    results = EntityFromList(seq=seq, scheme=IOB2).entities
    return results


def logging(passage, results):
    logger.info(f"passage: {passage}")
    logger.info("==== Top 1 prediction ====")
    for ent in sorted(results, key=lambda x: x.start_pos):
        logger.info(ent)


def testcase_one(model, testcase1):
    passage = testcase1["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_two(model, testcase2):
    passage = testcase2["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_three(model, testcase3):
    passage = testcase3["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_four(model, testcase4):
    passage = testcase4["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_five(model, testcase5):
    passage = testcase5["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_six(model, testcase6):
    passage = testcase6["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_seven(model, testcase7):
    passage = testcase7["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_eight(model, testcase8):
    passage = testcase8["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_nine(model, testcase9):
    passage = testcase9["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_ten(model, testcase10):
    passage = testcase10["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_eleven(model, testcase11):
    passage = testcase11["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_twelve(model, testcase12):
    passage = testcase12["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_thirteen(model, testcase13):
    passage = testcase13["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_fourteen(model, testcase14):
    passage = testcase14["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_fifteen(model, testcase15):
    passage = testcase15["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_sixteen(model, testcase16):
    passage = testcase16["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_seventeen(model, testcase17):
    passage = testcase17["passage"]
    results = predict(model, passage)
    logging(passage, results)


def testcase_eighteen(model, testcase18):
    passage = testcase18["passage"]
    results = predict(model, passage)
    logging(passage, results)

