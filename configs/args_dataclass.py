# encoding=utf-8
# Author: Allen.Chiang
# Description: Definition of custom arguments

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the train, dev, and test set."
        }
    )
    labels_path: str = field(metadata={"help": "Path to a file containing all labels."})
    train_filename: str = field(
        default="train.txt", metadata={"help": "The filename of training set."}
    )
    dev_filename: str = field(
        default="dev.txt", metadata={"help": "The filename of dev set."}
    )
    test_filename: str = field(
        default="test.txt", metadata={"help": "The filename of test set."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )


@dataclass
class SLDataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class QASLDataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


@dataclass
class MRCDataArguments:
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
            "be truncated to this length."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False,
        metadata={
            "help": "If true, the SQuAD examples contain some that do not have an answer."
        },
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "If null_score - best_non_null is greater than the threshold predict null."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "If null_score - best_non_null is greater than the threshold predict null."
        },
    )
    lang_id: int = field(
        default=0,
        metadata={
            "help": "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
        },
    )
    threads: int = field(
        default=1,
        metadata={"help": "multiple threads for converting example to features"},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    with_bilstmcrf: bool = field(
        metadata{
            "help": "Whether to use bilstmcrf."
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    return_dict: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the model should return a :class: `~transformers.file_utils.ModelOutput` instead of a plain tuple."
            "It does not show in the BertConfig when it is False."
        },
    )
    class_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Set the list of float for weighted cross entropy. It's not essential to be add up to 1."
            "Num of list must be the same as num of labels which is from label_path."
            "Order of list must be the same as order of labels which is from label_path."
            "Ususally set it as inverse class count."
        },
    )
