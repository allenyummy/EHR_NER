# encoding=utf-8
# Author: Allen.Chiang
# Description: Test for config

import json
import logging
import pytest

logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def args():
    from transformers import HfArgumentParser, TrainingArguments
    from configs.args_dataclass import DataTrainingArguments, ModelArguments
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_json_file(json_file="configs/config.json")  #pylint: disable=unbalanced-tuple-unpacking
    return data_args, model_args, training_args


class TestArgsElements:
    def test_args_elements(self, args):
        data_args, model_args, training_args = args
        logger.debug(f"data_training_args: {data_args}")
        logger.debug(f"model_args: {model_args}")
        logger.debug(f"training_args: {training_args}")

        assert data_args.data_dir != None
        assert data_args.labels_path != None
        assert model_args.task in ["sl", "mrc"]
        assert model_args.model_name_or_path != None
        assert training_args.output_dir != None
        assert training_args.do_train or training_args.do_eval or training_args.do_predict


class TestNerAsSLDataset:
    def test_dataset(self):
        pass
        # raise NotImplementedError

    
    def test_func_read_examples_from_file(self):
        from utils.sl import read_examples_from_file
        
        
        
        # raise NotImplementedError

    def test_func_convert_examples_to_features(self):
        from utils.sl import NerAsSLDataset
        pass
        # raise NotImplementedError
    


