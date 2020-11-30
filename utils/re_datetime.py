# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Regular expression for datetime

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class Pattern(Enum):
    pat0 = r"(\d{3,4})(\/|\.|-)?(\d{1,2})(\/|\.|-)?(\d{1,2})"
    pat1 = (
        r"(民國|西元)?"
        + r"(\d{3,4})+"
        + r"(年|\/|\.|-)+"
        + r"([0-1]?[0-9])"
        + r"(月|\/|\.|-)+"
        + r"([0-3]?[0-9])"
        + r"(日)?"
        + r"([0-2]?[0-9])?"
        + r"(時|:)?"
        + r"([0-6]?[0-9])?"
        + r"(分)?"
    )
