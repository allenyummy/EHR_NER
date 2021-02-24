# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Store testcases for global use

import logging
import os
import json
import pytest

logger = logging.getLogger(__name__)


def read_testcase_flat():
    data_path = os.path.join("tests", "test_model_pred", "testcase", "flat_ner.json")
    with open(data_path, "r", encoding="utf-8") as fd:
        in_data = json.load(fd)
    return in_data


def read_testcase_nested():
    data_path = os.path.join("tests", "test_model_pred", "testcase", "nested_ner.json")
    with open(data_path, "r", encoding="utf-8") as fd:
        in_data = json.load(fd)
    return in_data


flat_data = read_testcase_flat()
nested_data = read_testcase_nested()


# --- FLAT DATA ---
@pytest.fixture(scope="session")
def testcase_flat_1():
    return flat_data["data"][0]


@pytest.fixture(scope="session")
def testcase_flat_2():
    return flat_data["data"][1]


@pytest.fixture(scope="session")
def testcase_flat_3():
    return flat_data["data"][2]


@pytest.fixture(scope="session")
def testcase_flat_4():
    return flat_data["data"][3]


@pytest.fixture(scope="session")
def testcase_flat_5():
    return flat_data["data"][4]


@pytest.fixture(scope="session")
def testcase_flat_6():
    return flat_data["data"][5]


@pytest.fixture(scope="session")
def testcase_flat_7():
    return flat_data["data"][6]


@pytest.fixture(scope="session")
def testcase_flat_8():
    return flat_data["data"][7]


@pytest.fixture(scope="session")
def testcase_flat_9():
    return flat_data["data"][8]


@pytest.fixture(scope="session")
def testcase_flat_10():
    return flat_data["data"][9]


# --- NESTED DATA ---
@pytest.fixture(scope="session")
def testcase_nested_1():
    return nested_data["data"][0]


@pytest.fixture(scope="session")
def testcase_nested_2():
    return nested_data["data"][1]


@pytest.fixture(scope="session")
def testcase_nested_3():
    return nested_data["data"][2]


@pytest.fixture(scope="session")
def testcase_nested_4():
    return nested_data["data"][3]


@pytest.fixture(scope="session")
def testcase_nested_5():
    return nested_data["data"][4]


@pytest.fixture(scope="session")
def testcase_nested_6():
    return nested_data["data"][5]


@pytest.fixture(scope="session")
def testcase_nested_7():
    return nested_data["data"][6]


@pytest.fixture(scope="session")
def testcase_nested_8():
    return nested_data["data"][7]


@pytest.fixture(scope="session")
def testcase_nested_9():
    return nested_data["data"][8]


@pytest.fixture(scope="session")
def testcase_nested_10():
    return nested_data["data"][9]
