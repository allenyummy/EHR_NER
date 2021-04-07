# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: core class for sequence labeling

import logging
from enum import Flag, auto
from typing import List, NamedTuple, Tuple
import pandas as pd
from utils.seqhelper.src.scheme import Prefix, TypeRelation, IOB2

logger = logging.getLogger(__name__)


class EntityElements(NamedTuple):
    pid: int
    type: str
    start_pos: int
    end_pos: int
    text: str


class EntityFromNestedList:
    """Entity From Nested List"""

    def __init__(
        self, seqs: List[List[Tuple[str, str]]], scheme: str, delimiter: str = "-"
    ):
        logger.info(f"total passage: {len(seqs)}")
        self.entities = [
            EntityFromList(seq, scheme, delimiter=delimiter, pid=pid).entities
            for pid, seq in enumerate(seqs)
        ]

    def chunks2df(self):
        df = pd.DataFrame()
        for entities in self.entities:
            temp_df = pd.DataFrame(entities)
            df = df.append(temp_df, ignore_index=True)
            del temp_df
        return df


class EntityFromList:
    def __init__(
        self,
        seq: List[Tuple[str, str]],
        scheme,
        delimiter: str = "-",
        pid: int = 0,
    ):
        self.seq = seq
        self.extend_seq = self.seq + [("", "O")]
        self.scheme = scheme
        self.delimiter = delimiter
        self.pid = pid
        self.entities = self.get_entities()

    def get_entities(self) -> List[EntityElements]:
        prev_tag = "O"
        prev_type = ""
        begin_offset = 0
        chunks = list()
        for idx, chunk in enumerate(self.extend_seq):
            cur_label = chunk[1]
            cur_tag = cur_label[0]
            cur_type = cur_label[1:].split(self.delimiter, maxsplit=1)[-1]

            if Prefix[cur_tag] not in self.scheme.allowed_prefix:
                logger.debug(
                    f"{idx}-{chunk}: {Prefix[cur_tag]} is not available in {self.scheme.allowed_prefix}"
                )

            cur_pattern = (
                Prefix[prev_tag],
                Prefix[cur_tag],
                TypeRelation.SAME if prev_type == cur_type else TypeRelation.DIFF,
            )

            if trans_cur_tag_valid(self.scheme, cur_pattern, check_only=True):
                logger.debug(
                    f"Unreasonable transition. [pid:{self.pid}] [idx:{idx}] [{prev_tag}-{prev_type} =>> {cur_label} ({chunk[0]})]"
                )

            if is_pre_chunk_end(self.scheme, cur_pattern):
                chunk_text = "".join(
                    [token for token, _ in self.extend_seq[begin_offset:idx]]
                )
                chunk = EntityElements(
                    self.pid, prev_type, begin_offset, idx - 1, chunk_text
                )
                chunks.append(chunk)

            if is_cur_chunk_start(self.scheme, cur_pattern):
                begin_offset = idx

            prev_tag = cur_pattern[1]._name_
            prev_type = cur_type

        return chunks


def is_pre_chunk_end(scheme, cur_pattern: Tuple[Prefix, Prefix, TypeRelation]) -> bool:
    if cur_pattern in scheme.is_pre_chunk_end_patterns:
        return True
    return False


def is_cur_chunk_start(
    scheme, cur_pattern: Tuple[Prefix, Prefix, TypeRelation]
) -> bool:
    if cur_pattern in scheme.is_cur_chunk_start_patterns:
        return True
    return False


def trans_cur_tag_valid(
    scheme, cur_pattern: Tuple[Prefix, Prefix, TypeRelation], check_only: bool = True
) -> bool:
    if cur_pattern in scheme.trans_patterns:
        return True if check_only else scheme.trans_patterns[cur_pattern]
    return False if check_only else cur_pattern
