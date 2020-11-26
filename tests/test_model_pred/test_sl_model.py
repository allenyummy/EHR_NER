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


def testcase1(model, testcase16):
    passage = testcase16["passage"]
    results = model.predict(passage)
    token, label, prob = zip(*results)
    seq = [(t, l) for t, l in zip(token, label)]
    logger.info(f"passage: {passage}")
    for ent in EntityFromList(seq=seq, scheme=IOB2).entities:
        logger.info(ent)
