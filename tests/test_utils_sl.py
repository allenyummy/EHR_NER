# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for utils/sl.py

import os
import logging
import pytest
from utils.sl import (
    NerAsSLDataset,
    read_examples_from_file,
    convert_examples_to_features,
    get_labels,
)
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def data():
    data_dir = os.path.join("tests", "data", "sl")
    data_file = "fake.txt"
    label_file = "labels.txt"
    return {"data_dir": data_dir, "data_file": data_file, "label_file": label_file}


@pytest.fixture(scope="function")
def model():
    model_name = "hfl/chinese-bert-wwm"
    max_seq_length = 512
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return {
        "model_name": model_name,
        "max_seq_length": max_seq_length,
        "tokenizer": tokenizer,
    }


def test_get_labels(data):
    label_file_path = os.path.join(data["data_dir"], data["label_file"])
    labels, label_map, num_labels = get_labels(label_file_path)
    expected_label_map = {
        0: "O",
        1: "B-ADD",
        2: "I-ADD",
        3: "B-DCD",
        4: "I-DCD",
        5: "B-SGN",
        6: "I-SGN",
        7: "B-DTN",
        8: "I-DTN",
        9: "B-ICD",
        10: "I-ICD",
        11: "B-OCD",
        12: "I-OCD",
        13: "B-IBD",
        14: "I-IBD",
        15: "B-OBD",
        16: "I-OBD",
        17: "B-IND",
        18: "I-IND",
        19: "B-OND",
        20: "I-OND",
        21: "B-OPC",
        22: "I-OPC",
        23: "B-EMC",
        24: "I-EMC",
        25: "B-EMDE",
        26: "I-EMDE",
        27: "B-EMDS",
        28: "I-EMDS",
        29: "B-OPDE",
        30: "I-OPDE",
        31: "B-OPDS",
        32: "I-OPDS",
        33: "B-RTDE",
        34: "I-RTDE",
        35: "B-RTDS",
        36: "I-RTDS",
        37: "B-SGDE",
        38: "I-SGDE",
        39: "B-SGDS",
        40: "I-SGDS",
        41: "B-CTC",
        42: "I-CTC",
        43: "B-CTDE",
        44: "I-CTDE",
        45: "B-CTDS",
        46: "I-CTDS",
        47: "B-RTC",
        48: "I-RTC",
        49: "B-DIN",
        50: "I-DIN",
        51: "B-CTD",
        52: "I-CTD",
        53: "B-DPN",
        54: "I-DPN",
        55: "B-EMD",
        56: "I-EMD",
        57: "B-OPD",
        58: "I-OPD",
        59: "B-RTD",
        60: "I-RTD",
        61: "B-SGC",
        62: "I-SGC",
        63: "B-SGD",
        64: "I-SGD",
    }
    assert num_labels == len(expected_label_map)
    assert labels == list(expected_label_map.values())
    assert label_map == expected_label_map


def test_read_examples_from_file(data):
    data_file_path = os.path.join(data["data_dir"], data["data_file"])
    examples = read_examples_from_file(data_file_path)
    for example in examples:
        assert hasattr(example, "guid")
        assert hasattr(example, "words")
        assert hasattr(example, "labels")
        assert isinstance(example.guid, str)
        assert isinstance(example.words, list)
        assert isinstance(example.labels, list)


def test_convert_examples_to_features(data, model):
    data_file_path = os.path.join(data["data_dir"], data["data_file"])
    label_file_path = os.path.join(data["data_dir"], data["label_file"])
    examples = read_examples_from_file(data_file_path)
    labels, label_map, num_labels = get_labels(label_file_path)
    features = convert_examples_to_features(
        examples=examples,
        labels=labels,
        max_seq_length=model["max_seq_length"],
        tokenizer=model["tokenizer"],
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    )
    for feature in features:
        assert hasattr(feature, "input_ids")
        assert hasattr(feature, "attention_mask")
        assert hasattr(feature, "token_type_ids")
        assert hasattr(feature, "label_ids")
        assert isinstance(feature.input_ids, list)
        assert isinstance(feature.attention_mask, list)
        assert isinstance(feature.token_type_ids, list)
        assert isinstance(feature.label_ids, list)
        assert len(feature.input_ids) == model["max_seq_length"]
        assert len(feature.attention_mask) == model["max_seq_length"]
        assert len(feature.token_type_ids) == model["max_seq_length"]
        assert len(feature.label_ids) == model["max_seq_length"]


def test_NerAsSLDataset(data, model):
    label_file_path = os.path.join(data["data_dir"], data["label_file"])
    labels, label_map, num_labels = get_labels(label_file_path)
    dataset = NerAsSLDataset(
        data_dir=data["data_dir"],
        filename=data["data_file"],
        tokenizer=model["tokenizer"],
        labels=labels,
        model_type="bert",
        max_seq_length=model["max_seq_length"],
    )
    assert dataset.__len__() == len(dataset.features)
