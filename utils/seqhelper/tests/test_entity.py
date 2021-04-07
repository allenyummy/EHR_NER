# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: Test for src/entity.py

import logging
import pytest
from src.scheme import IOB2, Prefix, TypeRelation
from src.entity import (
    EntityFromList,
    EntityFromNestedList,
    is_pre_chunk_end,
    is_cur_chunk_start,
    trans_cur_tag_valid
)
from src.eval import (
    f1_score,
    precision_score,
    recall_score
)

logger = logging.getLogger(__name__)


test_IOB2_is_pre_chunk_end_data = [
    (IOB2, (Prefix.B, Prefix.B, TypeRelation.SAME), True),
    (IOB2, (Prefix.B, Prefix.B, TypeRelation.DIFF), True),
    (IOB2, (Prefix.B, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.B, Prefix.I, TypeRelation.DIFF), True),
    (IOB2, (Prefix.B, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.B, Prefix.O, TypeRelation.DIFF), True),
    (IOB2, (Prefix.I, Prefix.B, TypeRelation.SAME), True),
    (IOB2, (Prefix.I, Prefix.B, TypeRelation.DIFF), True),
    (IOB2, (Prefix.I, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.I, Prefix.I, TypeRelation.DIFF), True),
    (IOB2, (Prefix.I, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.I, Prefix.O, TypeRelation.DIFF), True),
    (IOB2, (Prefix.O, Prefix.B, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.B, TypeRelation.DIFF), False),
    (IOB2, (Prefix.O, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.I, TypeRelation.DIFF), False),
    (IOB2, (Prefix.O, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.O, TypeRelation.DIFF), False),
]
test_IOB2_is_cur_chunk_start_data = [
    (IOB2, (Prefix.B, Prefix.B, TypeRelation.SAME), True),
    (IOB2, (Prefix.B, Prefix.B, TypeRelation.DIFF), True),
    (IOB2, (Prefix.B, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.B, Prefix.I, TypeRelation.DIFF), True),
    (IOB2, (Prefix.B, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.B, Prefix.O, TypeRelation.DIFF), False),
    (IOB2, (Prefix.I, Prefix.B, TypeRelation.SAME), True),
    (IOB2, (Prefix.I, Prefix.B, TypeRelation.DIFF), True),
    (IOB2, (Prefix.I, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.I, Prefix.I, TypeRelation.DIFF), True),
    (IOB2, (Prefix.I, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.I, Prefix.O, TypeRelation.DIFF), False),
    (IOB2, (Prefix.O, Prefix.B, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.B, TypeRelation.DIFF), True),
    (IOB2, (Prefix.O, Prefix.I, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.I, TypeRelation.DIFF), True),
    (IOB2, (Prefix.O, Prefix.O, TypeRelation.SAME), False),
    (IOB2, (Prefix.O, Prefix.O, TypeRelation.DIFF), False),
]
test_IOB2_trans_chunk_valid_data = [
    (IOB2, (Prefix.B, Prefix.I, TypeRelation.DIFF), False, (Prefix.B, Prefix.B, TypeRelation.DIFF)),
    (IOB2, (Prefix.I, Prefix.I, TypeRelation.DIFF), False, (Prefix.I, Prefix.B, TypeRelation.DIFF)),
    (IOB2, (Prefix.O, Prefix.I, TypeRelation.DIFF), False, (Prefix.O, Prefix.B, TypeRelation.DIFF)),
]
test_IOB2_entity_from_list = [
    ([("台", "B-LOC"), ("北", "I-LOC"), ("是", "O"), ("阿", "B-PER"), ("倫", "I-PER"), ("的", "O"), ("家", "O")], [(0, "LOC", 0, 1, "台北"), (0, "PER", 3, 4, "阿倫")]),
    ([("阿", "B-PER"),("倫", "I-PER"),("是", "O"),("人", "B-ANI")], [(0, "PER", 0, 1, "阿倫"), (0, "ANI", 3, 3, "人")])
]
test_IOB2_entity_from_nested_list = [
    ([[
        ("台", "B-LOC"), 
        ("北", "I-LOC"), 
        ("是", "O"), 
        ("阿", "B-PER"), 
        ("倫", "I-PER"), 
        ("的", "O"), 
        ("家", "O")],
      [
        ("阿", "B-PER"),
        ("倫", "I-PER"),
        ("是", "O"),
        ("人", "B-ANI")]
     ], 
     [[
        (0, "LOC", 0, 1, "台北"), 
        (0, "PER", 3, 4, "阿倫")],
      [
        (1, "PER", 0, 1, "阿倫"), 
        (1, "ANI", 3, 3, "人")]
     ]
    )
]
test_eval_IOB2_entity_from_nested_list = [
    ([[
        ("台", "B-LOC"), 
        ("北", "I-LOC"), 
        ("是", "O"), 
        ("阿", "B-PER"), 
        ("倫", "I-PER"), 
        ("的", "O"), 
        ("家", "O")],
      [
        ("阿", "B-PER"),
        ("倫", "I-PER"),
        ("是", "O"),
        ("人", "B-ANI")]
     ], 
     [[
        ("台", "B-LOC"), 
        ("北", "O"), 
        ("是", "O"), 
        ("阿", "B-PER"), 
        ("倫", "I-PER"), 
        ("的", "O"), 
        ("家", "O")],
      [
        ("阿", "B-PER"),
        ("倫", "I-PER"),
        ("是", "O"),
        ("人", "O")]
     ], 
     0.6667,
     0.5,
     0.5714,
    )
]



@pytest.mark.parametrize(
    "scheme, cur_pattern, expected_end",
    test_IOB2_is_pre_chunk_end_data,
    ids=lambda val: str(val)
)
def test_is_pre_chunk_end(scheme, cur_pattern, expected_end):
    assert is_pre_chunk_end(scheme, cur_pattern) == expected_end

@pytest.mark.parametrize(
    "scheme, cur_pattern, expected_start",
    test_IOB2_is_cur_chunk_start_data,
    ids=lambda val: str(val)
)
def test_is_cur_chunk_start(scheme, cur_pattern, expected_start):
    assert is_cur_chunk_start(scheme, cur_pattern) == expected_start

@pytest.mark.parametrize(
    "scheme, cur_pattern, check_only, expected_trans_pattern",
    test_IOB2_trans_chunk_valid_data,
    ids=lambda val: str(val)
)
def test_trans_cur_tag_valid(scheme, cur_pattern, check_only, expected_trans_pattern):
    assert trans_cur_tag_valid(scheme, cur_pattern, check_only) == expected_trans_pattern

@pytest.mark.parametrize(
    "seq, expected_entitiy",
    test_IOB2_entity_from_list,
)
def test_IBO2_entitiy_from_list(seq, expected_entitiy):
    entity = EntityFromList(seq, IOB2).entities
    assert entity == expected_entitiy

@pytest.mark.parametrize(
    "seqs, expected_entitiy",
    test_IOB2_entity_from_nested_list,
)
def test_IBO2_entitiy_from_nested_list(seqs, expected_entitiy):
    entity = EntityFromNestedList(seqs, IOB2).entities
    assert entity == expected_entitiy

@pytest.mark.parametrize(
    "trues, preds, expected_precision_score, expected_recall_score, expected_f1_score",
    test_eval_IOB2_entity_from_nested_list,
)
def test_eval_IBO2_entitiy_from_nested_list(trues, preds, expected_precision_score, expected_recall_score, expected_f1_score):
    p = precision_score(trues, preds, IOB2)
    r = recall_score(trues, preds, IOB2)
    f1 = f1_score(trues, preds, IOB2)
    assert round(p, 4) == expected_precision_score
    assert round(r, 4) == expected_recall_score
    assert round(f1, 4) == expected_f1_score