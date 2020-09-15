# coding=utf-8
# Author: Allen.Chiang
# Description: Test for config

import json
import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def config():
    with open("configs/config.json", "r") as f:
        return json.load(f)

def test_config_elements(config):
    assert config.get("data_dir") != None
    assert config.get("labels_path") != None
    assert config.get("task") in ["sl", "mrc"]
    assert config.get("model_name_or_path") != None
    assert config.get("output_dir") != None
    assert config.get("do_train") or config.get("do_eval") or config.get("do_predict")
