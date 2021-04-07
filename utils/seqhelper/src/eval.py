# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: evaluation for sequence labeling

import logging
import numpy as np
from typing import List, Tuple
from collections import defaultdict
from src.scheme import IOB2
from src.reporter import DictReporter, StringReporter
from src.entity import EntityFromList, EntityFromNestedList

logger = logging.getLogger(__name__)


def f1_score(y_true, y_pred, scheme, mode="micro"):
    """Compute F1 score."""
    nb_correct, nb_pred, nb_true = _calculate_overall(y_true, y_pred, scheme, mode)
    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    return score


def precision_score(y_true, y_pred, scheme, mode="micro"):
    """Compute precision score."""
    nb_correct, nb_pred, _ = _calculate_overall(y_true, y_pred, scheme, mode)
    score = nb_correct / nb_pred if nb_pred > 0 else 0
    return score


def recall_score(y_true, y_pred, scheme, mode="micro"):
    """Compute recall score."""
    nb_correct, _, nb_true = _calculate_overall(y_true, y_pred, scheme, mode)
    score = nb_correct / nb_true if nb_true > 0 else 0
    return score


def accuracy_score(y_true, y_pred, scheme):
    """Compute accuracy score."""
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)
    score = nb_correct / nb_true
    return score


def classification_report(y_true, y_pred, scheme, digits: int=2, return_dict: bool=False):
    """Generate classification report."""
    all_type, nb_correct, nb_pred, nb_true = _calculate_each(y_true, y_pred, scheme)
    nb_correct = np.array(nb_correct)
    nb_pred = np.array(nb_pred)
    nb_true = np.array(nb_true)

    ## each
    p = nb_correct / nb_pred
    p[p == np.inf] = 0
    r = nb_correct / nb_true
    r[r == np.inf] = 0
    f = 2 * p * r / (p + r)
    f[f == np.inf] = 0
    s = nb_true

    ## micro
    micro_p = nb_correct.sum() / nb_pred.sum() if nb_pred.sum() > 0 else 0
    micro_r = nb_correct.sum() / nb_true.sum() if nb_true.sum() > 0 else 0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r > 0 else 0
    support = nb_true.sum()

    ## macro
    macro_p = p.sum()/len(p)
    macro_r = r.sum()/len(r)
    macro_f = f.sum()/len(f)

    if return_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, all_type))
        avg_width = len('weighted avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    for row in zip(all_type, p, r, f, s):
        reporter.write(*row)
    reporter.write_blank()
    reporter.write("micro avg", micro_p, micro_r, micro_f, support)
    reporter.write("macro avg", macro_p, macro_r, macro_f, support)
    reporter.write_blank()

    return reporter.report()

def _calculate_overall(y_true, y_pred, scheme, mode="micro"):
    true_entities, pred_entities = _toSet(y_true, y_pred, scheme)
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)
    return nb_correct, nb_pred, nb_true


def _calculate_each(y_true, y_pred, scheme):
    t, p = _toSet(y_true, y_pred, scheme)
    true_entities = defaultdict(set)
    pred_entities = defaultdict(set)
    for e in t:
        true_entities[e.type].add((e.pid, e.start_pos, e.end_pos, e.text))
    for e in p:
        pred_entities[e.type].add((e.pid, e.start_pos, e.end_pos, e.text))
    
    all_type = sorted(set(true_entities.keys()) | set(pred_entities.keys()))
    nb_correct = []
    nb_pred = []
    nb_true = []
    for type in all_type:
        temp_t = true_entities.get(type, set())
        temp_p = pred_entities.get(type, set())
        nb_correct.append(len(temp_t & temp_p))
        nb_pred.append(len(temp_p))
        nb_true.append(len(temp_t))
    return all_type, nb_correct, nb_pred, nb_true


def _toSet(y_true, y_pred, scheme):
    if any(isinstance(s, list) for s in y_true) and any(isinstance(s, list) for s in y_pred):
        true_entities = set()
        t = EntityFromNestedList(y_true, scheme).entities
        for l in t:
            for e in l:
                true_entities.add(e)
        pred_entities = set()
        p = EntityFromNestedList(y_pred, scheme).entities
        for l in p:
            for e in l:
                pred_entities.add(e)
    elif isinstance(y_true, list) and isinstance(y_pred, list):
        true_entities = set(EntityFromList(y_true, scheme).entities)
        pred_entities = set(EntityFromList(y_pred, scheme).entities)
    else:
        logger.error(
            "Please check if y_true and y_pred are same types. (list or nested list)"
        )
    return true_entities, pred_entities
