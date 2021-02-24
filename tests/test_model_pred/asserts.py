# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Utils for tests functions

import logging
from typing import List, Dict, NamedTuple

logger = logging.getLogger(__name__)


def trans(ans: List[Dict], ents: List[NamedTuple]):
    ans_set = set()
    for a in sorted(ans, key=lambda x: (x["start_pos"], x["end_pos"], x["type"])):
        if a["type"] != "DIN":
            ans_set.add((a["type"], a["start_pos"], a["end_pos"]))
    ents_set = set()
    for r in sorted(ents, key=lambda x: (x.start_pos, x.end_pos, x.type)):
        if r.type != "DIN":
            ents_set.add((r.type, r.start_pos, r.end_pos))
    return ans_set, ents_set


def assertExactMatch(ans, ents):
    ans_s, ents_s = trans(ans, ents)
    assert isinstance(ans_s, set)
    assert isinstance(ents_s, set)
    assert ans_s == ents_s
