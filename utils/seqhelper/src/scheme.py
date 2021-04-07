# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: scheme for sequence labeling

import logging
from enum import Flag, auto
from typing import List, Tuple

logger = logging.getLogger(__name__)


class Prefix(Flag):
    B = auto()
    I = auto()
    E = auto()
    O = auto()
    S = auto()
    U = auto()
    L = auto()
    ANY = B | I | E | O | S | U | L

class TypeRelation(Flag):
    SAME = auto()
    DIFF = auto()

class IOB2:
    allowed_prefix = Prefix.I | Prefix.O | Prefix.B
    is_cur_chunk_start_patterns = {
        (Prefix.B, Prefix.B, TypeRelation.SAME),
        (Prefix.B, Prefix.B, TypeRelation.DIFF),
        (Prefix.B, Prefix.I, TypeRelation.DIFF),
        (Prefix.I, Prefix.B, TypeRelation.SAME),
        (Prefix.I, Prefix.B, TypeRelation.DIFF),
        (Prefix.I, Prefix.I, TypeRelation.DIFF),
        (Prefix.O, Prefix.B, TypeRelation.DIFF),
        (Prefix.O, Prefix.I, TypeRelation.DIFF),
    }
    is_pre_chunk_end_patterns = {
        (Prefix.B, Prefix.B, TypeRelation.SAME),
        (Prefix.B, Prefix.B, TypeRelation.DIFF),
        (Prefix.B, Prefix.I, TypeRelation.DIFF),
        (Prefix.B, Prefix.O, TypeRelation.DIFF),
        (Prefix.I, Prefix.B, TypeRelation.SAME),
        (Prefix.I, Prefix.B, TypeRelation.DIFF),
        (Prefix.I, Prefix.I, TypeRelation.DIFF),
        (Prefix.I, Prefix.O, TypeRelation.DIFF),
    }
    available_patterns = {
        (Prefix.B, Prefix.I, TypeRelation.SAME),
        (Prefix.I, Prefix.I, TypeRelation.SAME),
        (Prefix.O, Prefix.O, TypeRelation.SAME),
    }
    trans_patterns = {
        (Prefix.B, Prefix.I, TypeRelation.DIFF): (Prefix.B, Prefix.B, TypeRelation.DIFF),
        (Prefix.I, Prefix.I, TypeRelation.DIFF): (Prefix.I, Prefix.B, TypeRelation.DIFF),
        (Prefix.O, Prefix.I, TypeRelation.DIFF): (Prefix.O, Prefix.B, TypeRelation.DIFF),
    }
    impossible_patterns = {
        (Prefix.B, Prefix.O, TypeRelation.SAME),
        (Prefix.I, Prefix.O, TypeRelation.SAME),
        (Prefix.O, Prefix.B, TypeRelation.SAME),
        (Prefix.O, Prefix.I, TypeRelation.SAME),
        (Prefix.O, Prefix.O, TypeRelation.DIFF),
    }

