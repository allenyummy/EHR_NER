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
import torch
from torch import nn

# from torch.utils.tensorboard import SummaryWriter
from configs.args_dataclass import (
    DataArguments,
    SLDataArguments,
    QASLDataArguments,
    MRCDataArguments,
    ModelArguments,
)
from utils.metrics_sl import accuracy_score, f1_score, precision_score, recall_score
from utils.sl import (
    NerAsSLDataset,
    get_labels,
    write_predictions_to_file,
)
from utils.qasl import NerAsQASLDataset
from utils.mrc import NerAsMRCDataset
from models.bert_sl import BertSLModel
from models.bertbilstmcrf_sl import BertBiLSTMCRFSLModel
from models.bert_qasl import BertQASLModel
from models.bertbilstmcrf_qasl import BertBiLSTMCRFQASLModel
from models.bert_mrc import BertMRCModel

logging.config.fileConfig("configs/logging.conf")
logger = logging.getLogger(__name__)


def align_predictions(
    predictions: np.ndarray, label_ids: np.ndarray
) -> Tuple[List[List[str]], List[List[str]]]:

    # --- Argmax to get best prediction ---
    preds = np.argmax(predictions, axis=2)

    # --- Drop nn.CrossEntropyLoss().ignore_index ---
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
    return out_label_list, preds_list


def compute_metrics(p: EvalPrediction) -> Dict:
    out_label_list, preds_list = align_predictions(p.predictions, p.label_ids)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def align_predictions_crf(
    predictions: np.ndarray, label_ids: np.ndarray
) -> Tuple[List[List[str]], List[List[str]]]:

    # --- Construct Attention Mask from Labelid first ---
    # seqs:      [CLS], a, b, c, [SEP], [PAD], [PAD]
    # label_ids:  -100, ., ., .,  -100,  -100,  -100
    # mask:          1, 1, 1, 1,     1,     0,     0

    # seqs:      [CLS], a, b, c, [SEP], d, e, f, [PAD], [PAD]
    # label_ids:  -100, ., ., .,  -100, ., ., .,  -100,  -100
    # mask:          1, 1, 1, 1,     1, 1, 1, 1,     0,     0

    batch_size, seq_len, num_labels = predictions.shape
    attention_masks = []
    for i in range(batch_size):
        locs = np.where(label_ids[i] == nn.CrossEntropyLoss().ignore_index)[0]
        pad_loc = np.split(locs, np.cumsum(np.where(locs[1:] - locs[:-1] > 1)) + 1)[-1][1:]
        attention_mask = [1] * (len(label_ids[i]) - len(pad_loc)) + [0] * len(pad_loc)
        attention_masks.append(attention_mask)

    # --- Decode best path by CRF ---
    emissions = torch.from_numpy(predictions).cuda()
    attention_masks = torch.FloatTensor(attention_masks).cuda()
    best_path = trainer.model.crf.decode(
        emissions=emissions, mask=attention_masks.type(torch.uint8)
    )

    # --- Drop nn.CrossEntropyLoss().ignore_index ---
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[best_path[i][j]])
    return out_label_list, preds_list


def compute_metrics_crf(p: EvalPrediction) -> Dict:
    out_label_list, preds_list = align_predictions_crf(p.predictions, p.label_ids)
    return {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }


def main():
    # --- global variable ---
    global label_map
    global trainer

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
    add_tokens = ["瘜","皰","搐","齲","蛀","髕","闌","疝","嚥","簍",
                  "廔","顳","溼","髖","膈","搔","攣","仟","鐙","蹠","橈"]
    tokenizer.add_tokens(add_tokens)
    logger.debug(f"Add tokens: {add_tokens}")

    # --- NER as Sequence Labeling Task ---
    if sys.argv[1] == "sl":

        # --- Prepare model ---
        logger.info("======= Prepare model =======")
        if model_args.with_bilstmcrf:
            logger.info("Init BertBiLSTMCRFSLModel")
            model = BertBiLSTMCRFSLModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Init BertBSLModel")
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
            compute_metrics=compute_metrics_crf
            if model_args.with_bilstmcrf
            else compute_metrics,
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
            if model_args.with_bilstmcrf:
                _, preds_list = align_predictions_crf(predictions, label_ids)
            else:
                _, preds_list = align_predictions(predictions, label_ids)

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
        if model_args.with_bilstmcrf:
            logger.info("Init BertBiLSTMCRFQASLModel")
            model = BertBiLSTMCRFQASLModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                class_weights=model_args.class_weights,  ## knowhow class weights
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Init BertQASLModel")
            model = BertQASLModel.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                class_weights=model_args.class_weights,  ## knowhow class weights
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
            compute_metrics=compute_metrics_crf
            if model_args.with_bilstmcrf
            else compute_metrics,
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
