# encoding=utf-8
# Author: Allen.Chiang
# Description: argsd of sequence labeling

from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the train, dev, and test set."}
    )
    labels_path: str = field(
        metadata={"help": "Path to a file containing all labels."}
    )
    train_filename: str = field(
        default="train.txt",
        metadata={"help": "The filename of training set."}
    )
    dev_filename: str = field(
        default="dev.txt",
        metadata={"help": "The filename of dev set."}
    )
    test_filename: str = field(
        default="test.txt",
        metadata={"help": "The filename of test set."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        }
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    task: str = field(
        metadata={"help": "It takes NER either as sl (sequence labeling) or mrc (machine reading comprehension)."}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(
        default=False, 
        metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )