# encoding=utf-8
# Author: Allen.Chiang
# Description: utils of machine reading comprehension dataset

import logging
import os
import json
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer
import torch
from torch import nn
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    qas_id: str
    question_text: str
    passage_text: str
    passage_tokens: List[str]
    answer_text_list: Optional[List[str]]
    start_pos: Optional[List[int]]
    end_pos: Optional[List[int]]
    ner_category: Optional[str]
    is_impossible: Optional[bool]


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    start_positions: Optional[List[int]]
    end_positions: Optional[List[int]]


class NerAsMRCDataset(Dataset):

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        filename: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
    ):
        self.set = filename.split(".")[0]

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                self.set, tokenizer.__class__.__name__, str(max_seq_length)
            ),
        )

        # Make sure only the first process in distributed training process the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(
                    f"Loading features from cache file at {cached_features_file}"
                )
                self.features = torch.load(cached_features_file)
            else:
                file_path = os.path.join(data_dir, filename)
                logger.info(f"Creating features from dataset file at {file_path}")
                examples = read_examples_from_file(file_path)
                self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
                    cls_token_at_end=bool(
                        model_type in ["xlnet"]
                    ),  # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    sep_token_extra=False,  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(file_path: str) -> List[InputExample]:
    with open(file_path, "r", encoding="utf-8") as f:
        mrc_data = json.load(f)
    version = mrc_data["version"]
    query = mrc_data["query"]
    data = mrc_data["data"]

    logger.info(f"data version: {version}")
    logger.info(f"queries: {query}")
    logger.info(f"data size: {len(data)}")

    examples = []
    for eachData in data:
        pid = eachData["pid"]
        passage = eachData["passage"]
        passage_tokens = eachData["passage_tokens"]

        for qid, (ner_cate, question) in enumerate(zip(query.keys(), query.values())):
            example = InputExample(
                qas_id=f"{pid}-{qid}",
                question_text=question,
                passage_text=passage,
                passage_tokens=passage_tokens,
                answer_text_list=[],
                start_pos=[],
                end_pos=[],
                ner_category=ner_cate,
                is_impossible=True,
            )

            for ans in eachData["answers"]:
                ans_text = ans["text"]
                label = ans["label"]
                start_pos = ans["start_pos"]
                end_pos = ans["end_pos"]

                if ner_cate in label:
                    example.answer_text_list.append(ans_text)
                    example.start_pos.append(start_pos)
                    example.end_pos.append(end_pos)
                    example.is_impossible = False

            examples.append(example)
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    labels: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """Loads a data file into a list of `InputFeatures`
    `cls_token_at_end` define the location of the CLS token:
        - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(labels)}

    features = []
    for (ex_index, example) in enumerate(examples):

        # --- init ---
        tokens = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        start_positions = []
        end_positions = []

        # --- process passage, start_pos, end_pos ---
        passage_tokens = example.passage_tokens
        all_doc_tokens = []
        doc_start_pos = []
        doc_end_pos = []

        if example.is_impossible:
            if not example.start_pos and not example.end_pos:
                doc_start_pos = []
                doc_end_pos = []
                all_doc_tokens = []
                for token in passage_tokens:
                    sub_token = tokenizer.tokenize(token)
                    all_doc_tokens.extend(sub_token)
                doc_start_pos = [0] * len(all_doc_tokens)
                doc_end_pos = [0] * len(all_doc_tokens)
            else:
                raise ValueError("Please check the answer: {example}")
        else:
            fake_start_pos = [0] * len(passage_tokens)
            fake_end_pos = [0] * len(passage_tokens)

            for start in example.start_pos:
                fake_start_pos[start] = 1
            for end in example.end_pos:
                fake_end_pos[end] = 1

            for token, start, end in zip(passage_tokens, fake_start_pos, fake_end_pos):
                sub_token = tokenizer.tokenize(token)

                if len(sub_token) > 1:
                    if start == 0:
                        doc_start_pos.extend([0] * len(sub_token))
                    if end == 0:
                        doc_end_pos.extend([0] * len(sub_token))
                    if start != 0:
                        doc_start_pos.append(1)
                        doc_start_pos.extend([0] * (len(sub_token) - 1))
                    if end != 0:
                        doc_end_pos.extend([0] * (len(sub_token) - 1))
                        doc_end_pos.append(1)
                    all_doc_tokens.extend(sub_token)

                elif len(sub_token) == 1:
                    doc_start_pos.append(start)
                    doc_end_pos.append(end)
                    all_doc_tokens.extend(sub_token)

                else:
                    raise ValueError("Please check the result of tokenizer !!! !!! ")

            assert len(all_doc_tokens) == len(doc_start_pos)
            assert len(all_doc_tokens) == len(doc_end_pos)
            assert len(doc_start_pos) == len(doc_end_pos)

        # --- process query ---
        query_tokens = tokenizer.tokenize(example.question_text)

        # --- truncate passage if length of ([CLS] Q [SEP] P [SEP]) is over max_seq_length ---
        max_seq_length_for_doc_tokens = max_seq_length - len(query_tokens) - 3
        if len(all_doc_tokens) >= max_seq_length_for_doc_tokens:
            all_doc_tokens = all_doc_tokens[:max_seq_length_for_doc_tokens]
            doc_start_pos = doc_start_pos[:max_seq_length_for_doc_tokens]
            doc_end_pos = doc_end_pos[:max_seq_length_for_doc_tokens]

        # --- add query first ---
        tokens.extend(query_tokens)
        token_type_ids.extend([sequence_a_segment_id] * len(query_tokens))
        start_positions.extend([0] * len(query_tokens))
        end_positions.extend([0] * len(query_tokens))

        # --- add [SEP] token ---
        tokens.append("[SEP]")
        token_type_ids.append(sequence_a_segment_id)
        start_positions.append(0)
        end_positions.append(0)

        # --- add passage then ---
        tokens.extend(all_doc_tokens)
        token_type_ids.extend([sequence_b_segment_id] * len(all_doc_tokens))
        start_positions.extend(doc_start_pos)
        end_positions.extend(doc_end_pos)

        # --- add [SEP] tokens again ---
        tokens.append("[SEP]")
        token_type_ids.append(sequence_b_segment_id)
        start_positions.append(0)
        end_positions.append(0)

        # --- add [CLS] tokens ---
        if cls_token_at_end:
            tokens += [cls_token]
            token_type_ids += [cls_token_segment_id]
            start_positions += [0]
            end_positions += [0]
        else:
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            start_positions = [0] + start_positions
            end_positions = [0] + end_positions

        # --- transform to input ids ---
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # --- add attention mask if needed ---
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # --- add [PAD] tokens if needed ---
        padding_length = max_seq_length - len(input_ids)
        if padding_length > 0:
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
                start_positions = ([0] * padding_length) + start_positions
                end_positions = ([0] * padding_length) + end_positions
            else:
                input_ids += [pad_token] * padding_length
                attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
                token_type_ids += [pad_token_segment_id] * padding_length
                start_positions += [0] * padding_length
                end_positions += [0] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(start_positions) == max_seq_length
        assert len(end_positions) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("qas_id: %s", example.qas_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info(
                "start_positions: %s", " ".join([str(x) for x in start_positions])
            )
            logger.info("end_positions: %s", " ".join([str(x) for x in end_positions]))
            logger.info(
                "ner_category: %s (%d)",
                example.ner_category,
                label_map[example.ner_category],
            )
            logger.info("is_impossible: %s", example.is_impossible)

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )
        )
    return features


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


if __name__ == "__main__":

    from transformers import BertTokenizer

    file_path = "/Users/allenyummy/Documents/EHR_NER/data/mrc/dev.txt"
    label_path = "/Users/allenyummy/Documents/EHR_NER/data/mrc/labels.txt"
    labels = get_labels(label_path)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")

    examples = read_examples_from_file(file_path)

    features = convert_examples_to_features(
        examples,
        labels,
        max_seq_length=512,
        tokenizer=tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,  # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(tokenizer.padding_side == "left"),
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=tokenizer.pad_token_type_id,
    )
