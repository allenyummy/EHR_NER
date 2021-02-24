# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for api/bert_sl_predictor.py

import logging
import pytest
from api.bert_sl_predictor import BertSLPredictor
from asserts import assertExactMatch

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def model():
    model_dir = "trained_model/0817_8786_concat_num/sl/2020-09-02-00@hfl@chinese-bert-wwm@CE_S-512_B-4_E-100_LR-5e-5_SD-1/"
    model = BertSLPredictor(model_dir=model_dir, with_bilstmcrf=False)
    return model


def run_flat_test(model, testcase):
    passage = " ".join(testcase["passage_tokens"])
    flat_ans = testcase["flat_ne_answers"]  ## List[Dict]
    res = model.predict(passage)
    ents = model.refine(res)  ## List[NamedTuple]
    log(testcase, flat_ans, ents)
    assertExactMatch(flat_ans, ents)


def run_nested_test(model, testcase):
    passage = " ".join(testcase["passage_tokens"])
    nested_ans = testcase["nested_ne_answers"]  ## List[Dict]
    res = model.predict(passage)
    ents = model.refine(res)  ## List[NamedTuple]
    log(testcase, nested_ans, ents)
    assertExactMatch(nested_ans, ents)


def log(testcase, ans, ents):
    pid = testcase["pid"]
    passage = testcase["passage"]
    passage_tokens = testcase["passage_tokens"]
    logger.info(f"pid: {pid}")
    logger.info(f"passage: {passage}")

    logger.info("==== ANSWER ====")
    for a in sorted(ans, key=lambda x: (x["start_pos"], x["end_pos"], x["type"])):
        e_type = a["type"]
        if e_type != "DIN":
            start = a["start_pos"]
            end = a["end_pos"]
            text = "".join(passage_tokens[start : end + 1])
            logger.info(f"[{e_type}]-[{start},{end}]-[{text}]")

    logger.info("==== PREDICTION ====")
    for r in sorted(ents, key=lambda x: (x.start_pos, x.end_pos, x.type)):
        e_type = r.type
        if e_type != "DIN":
            start = r.start_pos
            end = r.end_pos
            text = "".join(passage_tokens[start : end + 1])
            logger.info(f"[{e_type}]-[{start},{end}]-[{text}]")


# --- TEST FOR FLAT NER ---
def test_flat_1(model, testcase_flat_1):
    run_flat_test(model, testcase_flat_1)


def test_flat_2(model, testcase_flat_2):
    run_flat_test(model, testcase_flat_2)


def test_flat_3(model, testcase_flat_3):
    run_flat_test(model, testcase_flat_3)


def test_flat_4(model, testcase_flat_4):
    run_flat_test(model, testcase_flat_4)


def test_flat_5(model, testcase_flat_5):
    run_flat_test(model, testcase_flat_5)


def test_flat_6(model, testcase_flat_6):
    run_flat_test(model, testcase_flat_6)


@pytest.mark.xfail(reason="DataBug: Passage Leads Model to Fail")
def test_flat_7(model, testcase_flat_7):
    run_flat_test(model, testcase_flat_7)


def test_flat_8(model, testcase_flat_8):
    run_flat_test(model, testcase_flat_8)


def test_flat_9(model, testcase_flat_9):
    run_flat_test(model, testcase_flat_9)


@pytest.mark.xfail(reason="ModelBug: Expected Single Span but Output Multi Spans")
def test_flat_10(model, testcase_flat_10):
    run_flat_test(model, testcase_flat_10)


# --- TEST FOR NESTED NER ---
@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_1(model, testcase_nested_1):
    run_nested_test(model, testcase_nested_1)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_2(model, testcase_nested_2):
    run_nested_test(model, testcase_nested_2)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_3(model, testcase_nested_3):
    run_nested_test(model, testcase_nested_3)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_4(model, testcase_nested_4):
    run_nested_test(model, testcase_nested_4)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_5(model, testcase_nested_5):
    run_nested_test(model, testcase_nested_5)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_6(model, testcase_nested_6):
    run_nested_test(model, testcase_nested_6)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_7(model, testcase_nested_7):
    run_nested_test(model, testcase_nested_7)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_8(model, testcase_nested_8):
    run_nested_test(model, testcase_nested_8)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_9(model, testcase_nested_9):
    run_nested_test(model, testcase_nested_9)


@pytest.mark.xfail(reason="ModelBug: Model is Designed For FlatNER, not NestedNER")
def test_nested_10(model, testcase_nested_10):
    run_nested_test(model, testcase_nested_10)
