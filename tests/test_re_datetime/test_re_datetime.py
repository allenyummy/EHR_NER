# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for utils/re_datetime.py

import logging
import pytest
import re
from utils.re_datetime import Pattern

logger = logging.getLogger(__name__)


def testcase_one(testcase2):
    passage = testcase2["passage"]
    pat = Pattern.pat1
    logger.info(passage)
    logger.info(f"{pat.value}")
    results = re.search(pat.value, passage)
    logger.info(results)
    logger.info(results.group())


def testcase_two():
    passage = "109##5##2吃飯"
    pat = Pattern.pat2
    logger.info(passage)
    logger.info(f"{pat.value}")
    results = re.search(pat.value, passage)
    logger.info(results)
    logger.info(results.group())