# encoding=utf-8
# Author: Allen.Chiang
# Description: run NER task
# sl: sequence labeling
# qasl: question answering sequence labeling
# mrc: machine reading comprehension

import logging
import logging.config
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from transformers import (
    BertConfig,
    BertTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    EvalPrediction,
)
from torch import nn

# from torch.utils.tensorboard import SummaryWriter
from configs.args_dataclass import (
    DataArguments,
    SLDataArguments,
    QASLDataArguments,
    MRCDataArguments,
    ModelArguments,
)
from utils.sl import (
    NerAsSLDataset,
    get_labels,
    write_predictions_to_file,
    align_predictions,
    compute_metrics,
)
from utils.qasl import NerAsQASLDataset
from utils.mrc import NerAsMRCDataset
from models.bert_sl import BertSLModel
from models.bert_qasl import BertQASLModel
from models.bert_mrc import BertMRCModel

logging.config.fileConfig("configs/logging.conf")
logger = logging.getLogger(__name__)


def main():

    # --- Parse args ---
    logger.info("======= Parse args =======")
    if len(sys.argv) != 2:
        raise ValueError(
            "Please enter the command: PYTHONPATH=./ python train/run_ner.py [sl, qasl, or mrc]"
        )

    if sys.argv[1] == "sl":
        logger.info("======= Traditional Sequence Labeling =======")
        parser = HfArgumentParser(
            (DataArguments, SLDataArguments, ModelArguments, TrainingArguments)
        )
    elif sys.argv[1] == "qasl":
        logger.info("======= QA Sequence Labeling =======")
        parser = HfArgumentParser(
            (DataArguments, QASLDataArguments, ModelArguments, TrainingArguments)
        )
    elif sys.argv[1] == "mrc":
        logger.info("======= Machine Reading Comprehension Sequence Labeling =======")
        parser = HfArgumentParser(
            (DataArguments, MRCDataArguments, ModelArguments, TrainingArguments)
        )
    else:
        raise ValueError(
            "The second argv of sys must be sl (sequence labeling), \
                                            qasl (QA sequence labeling), \
                                            mrc  (machine reading comprehension)."
        )

    config_json_file = os.path.join("configs", sys.argv[1] + "_config.json")
    data_args, task_args, model_args, training_args = parser.parse_json_file(
        json_file=config_json_file
    )

    logger.debug(f"data_args: {data_args}")
    logger.debug(f"task_args: {task_args}")
    logger.debug(f"model_args: {model_args}")
    logger.debug(f"training_args: {training_args}")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    set_seed(training_args.seed)

    # --- Prepare labels ---
    logger.info("======= Prepare labels =======")
    labels, label_map, num_labels = get_labels(data_args.labels_path)
    logger.debug(f"label_map: {label_map}")

    # --- Prepare model config ---
    logger.info("======= Prepare model config =======")
    config = BertConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        return_dict=model_args.return_dict,
    )
    logger.debug(f"config: {config}")

    # --- Prepare tokenizer ---
    logger.info("======= Prepare tokenizer =======")
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    # --- Add tokens that are might not in vocab.txt ---
    logger.info("======= Add tokens that are might not in vocab.txt =======")
    add_tokens = [
        "瘜",
        "皰",
        "搐",
        "齲",
        "蛀",
        "髕",
        "闌",
        "疝",
        "嚥",
        "簍",
        "廔",
        "顳",
        "溼",
        "髖",
        "膈",
        "搔",
        "攣",
        "仟",
        "鐙",
        "蹠",
        "橈",
    ]
    tokenizer.add_tokens(add_tokens)
    logger.debug(f"Add tokens: {add_tokens}")

    # --- NER as Sequence Labeling Task ---
    if sys.argv[1] == "sl":

        # --- Prepare model ---
        logger.info("======= Prepare model =======")
        model = BertSLModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model.resize_token_embeddings(len(tokenizer))

        # --- Prepare datasets ---
        logger.info("======= Prepare dataset =======")
        train_dataset = (
            NerAsSLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.train_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        eval_dataset = (
            NerAsSLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.dev_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        test_dataset = (
            NerAsSLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.test_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_eval or training_args.do_predict
            else None
        )

        # --- Initialize trainer from huggingface ---
        logger.info("======= Prepare trainer =======")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # tb_writer=SummaryWriter(training_args.logging_dir),
            # optimizers
        )

        # --- Train ---
        if training_args.do_train:
            trainer.train()
            trainer.save_model()
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)

        # --- Evaluate all set ---
        if training_args.do_eval:
            for dataset in [train_dataset, eval_dataset, test_dataset]:
                metrics = trainer.evaluate(dataset)
                evaluation_results_file = os.path.join(
                    training_args.output_dir, "evaluation_results.txt"
                )
                if trainer.is_world_master():
                    with open(evaluation_results_file, "a", encoding="utf-8") as writer:
                        logger.info(f"Eval results of {dataset.set} set")
                        writer.write(f"Eval results of {dataset.set} set\n")
                        for key, value in metrics.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
                        writer.write("\n")

        # --- Predict test set ---
        if training_args.do_predict:
            predictions, label_ids, metrics = trainer.predict(test_dataset)
            preds_list, _ = align_predictions(predictions, label_ids)
            test_predictions_file = os.path.join(
                training_args.output_dir, "test_predictions.txt"
            )
            if trainer.is_world_master():
                with open(test_predictions_file, "w", encoding="utf-8") as writer:
                    with open(
                        os.path.join(data_args.data_dir, data_args.test_filename),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        write_predictions_to_file(writer, f, preds_list)

    elif sys.argv[1] == "qasl":

        # --- Prepare model ---
        logger.info("======= Prepare model =======")
        model = BertQASLModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            class_weights=model_args.class_weights,
            cache_dir=model_args.cache_dir,
        )
        model.resize_token_embeddings(len(tokenizer))

        # --- Prepare datasets ---
        logger.info("======= Prepare dataset =======")
        train_dataset = (
            NerAsQASLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.train_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        eval_dataset = (
            NerAsQASLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.dev_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        test_dataset = (
            NerAsQASLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.test_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_eval or training_args.do_predict
            else None
        )

        # --- Initialize trainer from huggingface ---
        logger.info("======= Prepare trainer =======")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            # tb_writer=SummaryWriter(training_args.logging_dir),
            # optimizers
        )

        # --- Train ---
        if training_args.do_train:
            trainer.train()
            trainer.save_model()
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)

        # --- Evaluate all set ---
        if training_args.do_eval:
            for dataset in [train_dataset, eval_dataset, test_dataset]:
                metrics = trainer.evaluate(dataset)
                evaluation_results_file = os.path.join(
                    training_args.output_dir, "evaluation_results.txt"
                )
                if trainer.is_world_master():
                    with open(evaluation_results_file, "a") as writer:
                        logger.info(f"Eval results of {dataset.set} set")
                        writer.write(f"Eval results of {dataset.set} set\n")
                        for key, value in metrics.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
                        writer.write("\n")

    elif sys.argv[1] == "mrc":

        # --- Prepare model ---
        model = BertMRCModel.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model.resize_token_embeddings(len(tokenizer))

        # --- Prepare datasets ---
        train_dataset = (
            NerAsMRCDataset(
                data_dir=data_args.data_dir,
                filename=data_args.train_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        eval_dataset = (
            NerAsMRCDataset(
                data_dir=data_args.data_dir,
                filename=data_args.dev_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_train or training_args.do_eval
            else None
        )
        test_dataset = (
            NerAsMRCDataset(
                data_dir=data_args.data_dir,
                filename=data_args.test_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=task_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_eval or training_args.do_predict
            else None
        )

        # --- Initialize trainer from huggingface ---
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # --- Train ---
        if training_args.do_train:
            trainer.train()
            trainer.save_model()
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)

        # --- Predict test set ---
        if training_args.do_predict:
            predictions = trainer.predict(test_dataset)
            logger.info(f"preditcions: {predictions}")


if __name__ == "__main__":
    main()
