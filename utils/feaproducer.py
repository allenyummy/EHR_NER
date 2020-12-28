# encoding=utf-8
# Author: Yu-Lun Chiang
# Description: utils function to transform input json file to features as model input

import logging
import os
import json
from filelock import FileLock
from dataclasses import dataclass
from typing import List, Dict, Optional, TextIO
from transformers import PreTrainedTokenizer
import torch
from torch import nn
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    id: str
    passage_text: str
    passage_tokens: List[str]
    labels: Optional[List[str]]
    question_text: Optional[str] = None  ## be used when qasl


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class NerDataset(Dataset):

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        task: str,
        data_dir: str,
        filename: str,
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        max_seq_length: Optional[int] = None,
        query_path: Optional[str] = None,
        overwrite_cache=False,
    ):
        self.set = filename.split(".")[0]

        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                task, self.set, tokenizer.__class__.__name__, str(max_seq_length)
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                data_path = os.path.join(data_dir, filename)
                logger.info(f"Creating features from dataset file at {data_path}")
                query = None
                if "qasl" in task:
                    with open(query_path, "r", encoding="utf-8") as fq:
                        query = json.load(fq)
                examples = read_examples_from_file(data_path, query)
                # TODO clean up all this to leverage built-in features of tokenizers
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
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(data_path: str, query: Optional[Dict[str, str]]=None) -> List[InputExample]:
    # --- read data
    with open(data_path, "r", encoding="utf-8") as fd:
        in_data = json.load(fd)
    version = in_data["version"]
    data = in_data["data"]
    logger.info(f"data version: {version}")
    logger.info(f"data size: {len(data)}")

    # --- produce example for either sl or qasl
    examples = []
    for eachData in data:
        pid = eachData["pid"]
        passage = eachData["passage"]
        passage_tokens = eachData["passage_tokens"]
        
        # --- SL task
        if not query:
            example = InputExample(
                id=pid,
                passage_text=passage,
                passage_tokens=passage_tokens,
                labels=["O"] * len(passage_tokens),
            )
            anss = eachData["flat_ne_answers"]
            for ans in anss:
                type = ans["type"]
                text = ans["text"]
                start_pos = ans["start_pos"]
                end_pos = ans["end_pos"]
                assert text == "".join(passage_tokens[start_pos:end_pos+1])
                example.labels[start_pos] = f"B-{type}"
                example.labels[start_pos+1:end_pos+1] = [f"I-{type}"]*(end_pos-start_pos)
            examples.append(example)

        # --- QASL task
        else:
            anss = eachData["nested_ne_answers"]
            for qid, (ner_type, question) in enumerate(zip(query.keys(), query.values())):
                example = InputExample(
                    id=f"{pid}-{qid}",
                    question_text=question,
                    passage_text=passage,
                    passage_tokens=passage_tokens,
                    labels=["O"] * len(passage_tokens),
                )
                for ans in anss:
                    type = ans["type"]
                    text = ans["text"]
                    start_pos = ans["start_pos"]
                    end_pos = ans["end_pos"]
                    assert text == "".join(passage_tokens[start_pos:end_pos+1])
                    if ner_type == type:
                        example.labels[start_pos] = "B"
                        example.labels[start_pos+1:end_pos+1] = ["I"] * (end_pos-start_pos)
                examples.append(example)
    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    labels: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
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
        label_ids = []

        # --- process passage and label_ids ---
        doc_tokens = []
        doc_label_ids = []
        for passage_token, label in zip(example.passage_tokens, example.labels):
            sub_tokens = tokenizer.tokenize(passage_token)
            if len(sub_tokens) > 0:
                doc_tokens.extend(sub_tokens)
                doc_label_ids.extend(
                    [label_map[label]] + [pad_token_label_id] * (len(sub_tokens) - 1)
                )

        # --- process query ---
        query_tokens = []
        if example.question_text:
            query_tokens = tokenizer.tokenize(example.question_text)

        # --- truncate passage ---
        max_seq_length_for_doc_tokens = max_seq_length - len(query_tokens) - 3
        if len(doc_tokens) >= max_seq_length_for_doc_tokens:
            doc_tokens = doc_tokens[:max_seq_length_for_doc_tokens]
            doc_label_ids = doc_label_ids[:max_seq_length_for_doc_tokens]

        if not example.question_text:
            # --- add passage first ---
            tokens.extend(doc_tokens)
            token_type_ids.extend([sequence_a_segment_id] * len(doc_tokens))
            label_ids.extend(doc_label_ids)

            # --- add [SEP] tokens then ---
            tokens.append("[SEP]")
            token_type_ids.append(sequence_a_segment_id)
            label_ids.append(pad_token_label_id)
        
        else:
            # --- add query first ---
            tokens.extend(query_tokens)
            token_type_ids.extend([sequence_a_segment_id] * len(query_tokens))
            label_ids.extend([pad_token_label_id] * len(query_tokens))

            # --- add [SEP] token ---
            tokens.append("[SEP]")
            token_type_ids.append(sequence_a_segment_id)
            label_ids.append(pad_token_label_id)

            # --- add passage then ---
            tokens.extend(doc_tokens)
            token_type_ids.extend([sequence_b_segment_id] * len(doc_tokens))
            label_ids.extend(doc_label_ids)

            # --- add [SEP] tokens again ---
            tokens.append("[SEP]")
            token_type_ids.append(sequence_b_segment_id)
            label_ids.append(pad_token_label_id)

        # --- add [CLS] tokens ---
        if cls_token_at_end:
            tokens += [cls_token]
            token_type_ids += [cls_token_segment_id]
            label_ids += [pad_token_label_id]
        else:
            tokens = [cls_token] + tokens
            token_type_ids = [cls_token_segment_id] + token_type_ids
            label_ids = [pad_token_label_id] + label_ids

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
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
                token_type_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("id: %s", example.id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in attention_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in token_type_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
            )
        )
    return features


def write_predictions_to_file(
    writer: TextIO, test_input_reader: TextIO, preds_list: List
):
    example_id = 0
    for line in test_input_reader:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            writer.write(line)
            if not preds_list[example_id]:
                example_id += 1
        elif preds_list[example_id]:
            output_line = (
                line.split()[0]
                + " "
                + line.split()[1]
                + " "
                + preds_list[example_id].pop(0)
                + "\n"
            )
            writer.write(output_line)
        else:
            output_line = (
                line.split()[0]
                + " "
                + line.split()[1]
                + " "
                + "No prediciton due to limitation of max-seq-length"
                + "\n"
            )
            writer.write(output_line)
            logger.warning(
                "Maximum sequence length exceeded: No prediction for '%s'.",
                line.split()[0],
            )


def get_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


if __name__ == "__main__":

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")

    # --- sl
    data_path = "data/final/dev.json"
    label_path = "data/final/label/label4sl.txt"
    labels = get_labels(label_path)
    examples = read_examples_from_file(data_path)
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
    print (len(features))

    # --- qasl or simqasl
    data_path = "data/final/dev.json"
    label_path = "data/final/label/label4qasl.txt"
    labels = get_labels(label_path)
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm")
    query_path = "data/final/query/simqasl_query.json"
    with open(query_path, "r", encoding="utf-8") as fq:
        query = json.load(fq)

    examples = read_examples_from_file(data_path, query)
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
    print (len(features))