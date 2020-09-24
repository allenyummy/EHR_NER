# encoding=utf-8
# Author: Allen.Chiang
# Description: run NER task either as sequence labeling task or as machine reading comprehension

import logging
import logging.config
import os
import sys
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
    EvalPrediction
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from configs.args_dataclass import DataTrainingArguments, ModelArguments
from utils.sl import (
    NerAsSLDataset, 
    get_labels, 
    write_predictions_to_file,
)
from utils.mrc import NerAsMRCDataset

logging.config.fileConfig('configs/logging.conf')
logger = logging.getLogger(__name__)


def main():
    
    #--- Parse args ---#
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_json_file(json_file=sys.argv[1])  #pylint: disable=unbalanced-tuple-unpacking
    logger.debug(f"data_args: {data_args}")
    logger.debug(f"model_args: {model_args}")
    logger.debug(f"training_args: {training_args}")

    if model_args.task not in ["sl", "mrc"]:
        raise ValueError("task in configs/config.json must be either sl or mrc.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    set_seed(training_args.seed)

    #--- Prepare labels ---#
    labels = get_labels(data_args.labels_path)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    #--- Prepare model config and tokenizer ---#
    config = BertConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            id2label=label_map,
            label2id={label: i for i, label in enumerate(labels)},
            cache_dir=model_args.cache_dir,
        )
    tokenizer = BertTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    #--- Add tokens that are might not in vocab.txt ---#
    add_tokens = ['瘜', '皰', '搐', '齲', '蛀', '髕', '闌', '疝', '嚥', '簍', '廔', '顳', '溼', '髖', '膈', '搔', '攣', '仟', '鐙', '蹠', '橈']
    tokenizer.add_tokens(add_tokens)

    if model_args.task == "sl":

        #--- Prepare model ---#
        model = BertForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        model.resize_token_embeddings(len(tokenizer))

        #--- Prepare datasets ---#
        train_dataset = (
            NerAsSLDataset(
                data_dir=data_args.data_dir,
                filename=data_args.train_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
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
                max_seq_length=data_args.max_seq_length,
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
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_predict
            else None
        )
        
        #--- utils function for sl ---# (label_map is unique for sl task.)
        def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
            preds = np.argmax(predictions, axis=2)

            batch_size, seq_len = preds.shape

            out_label_list = [[] for _ in range(batch_size)]
            preds_list = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        out_label_list[i].append(label_map[label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])

            return preds_list, out_label_list

        def compute_metrics(p: EvalPrediction) -> Dict:
            preds_list, out_label_list = align_predictions(p.predictions, p.label_ids)
            return {
                "accuracy_score": accuracy_score(out_label_list, preds_list),
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1": f1_score(out_label_list, preds_list),
            }

        #--- Initialize trainer from huggingface ---#
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            tb_writer=SummaryWriter(training_args.logging_dir),
            # optimizers
        )

        #--- Train ---#
        if training_args.do_train:
            trainer.train()
            trainer.save_model()
            if trainer.is_world_master():
                tokenizer.save_pretrained(training_args.output_dir)
        
        #--- Evaluate all set ---#
        if training_args.do_eval:
            for dataset in [train_dataset, eval_dataset, test_dataset]:
                metrics = trainer.evaluate(dataset)
                evaluation_results_file = os.path.join(training_args.output_dir, "evaluation_results.txt")
                if trainer.is_world_master():
                    with open(evaluation_results_file, "w+") as writer:
                        logger.info(f"Eval results of {dataset.set}")
                        writer.write(f"Eval results of {dataset.set}\n")
                        for key, value in metrics.items():
                            logger.info("  %s = %s", key, value)
                            writer.write("%s = %s\n" % (key, value))
                        writer.write("\n")
        
        #--- Predict test set ---#
        if training_args.do_predict:
            predictions, label_ids, metrics = trainer.predict(test_dataset)
            preds_list, _ = align_predictions(predictions, label_ids)
            test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            if trainer.is_world_master():
                with open(test_predictions_file, "w") as writer:
                    with open(os.path.join(data_args.data_dir, data_args.test_filename), "r") as f:
                        write_predictions_to_file(writer, f, preds_list)
    
    elif model_args.task == "mrc":

        #--- Prepare datasets ---#
        train_dataset = (
            NerAsMRCDataset(
                data_dir=data_args.data_dir,
                filename=data_args.train_filename,
                tokenizer=tokenizer,
                labels=labels,
                model_type=config.model_type,
                max_seq_length=data_args.max_seq_length,
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
                max_seq_length=data_args.max_seq_length,
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
                max_seq_length=data_args.max_seq_length,
                overwrite_cache=data_args.overwrite_cache,
            )
            if training_args.do_predict
            else None
        )
        

if __name__ == "__main__":
    main()
    

    
    
    


    