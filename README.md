# Nested Named Entity Recognition for Chinese Electronic Health Records with QA-based Sequence Labeling
This study presents a novel QA-based sequence labeling (QASL) approach to naturally tackle both flat and nested Named Entity Recognition (NER) tasks on a Chinese Electronic Health Records (CEHRs) dataset. This proposed QASL approach parallelly asks a corresponding natural language question for each specific named entity type. It then identifies those associated NEs of the same specified type with the BIO tagging scheme. The associated nested NEs are then formed by overlapping the results of various types. Compared with those pure sequence-labeling (SL) approaches, since the given question includes significant prior knowledge about the specified entity type and the capability of extracting NEs with different types, the nested NER task is thus improved, obtaining 90.70% of F1-score. Besides, compared to the pure QA-based approach, our proposed approach retains the SL features, which could extract multiple NEs with the same types without knowing the exact number of NEs in the same passage in advance. Eventually, experiments on our CEHR dataset demonstrate that QASL-based models greatly outperform the SL-based models by 6.12% to 7.14% of F1-score.

You can check this paper [here](https://aclanthology.org/2021.rocling-1.3/).

## Typical Chinese Electronic Health Records 
![image](https://user-images.githubusercontent.com/36063123/135220330-f0d6cd3f-01e7-4131-ac9b-06a1f5b92ea2.png)

## Framework of QA-based Sequence Labeling (QASL)
![image](https://user-images.githubusercontent.com/36063123/135220369-a86c4dc3-34cd-4443-9698-7ee47d7bec4d.png)

---

## Demo
I've not owned a virtual private server to deploy the service. Therefore, I made a GIF to demostrate the operation of SL and QASL models. You can try by yourself by typing the following command.
```
$ PYTHONPATH=./ python demo/app.py
```

![demo](https://user-images.githubusercontent.com/36063123/114145365-e4c8ec00-9948-11eb-920a-86f5aff462a0.gif)

---
## Prepare Virtual Environment
+ Local virtual conda environment

You can build a local environment in your development environment.
```
$ git clone https://github.com/allenyummy/EHR_NER.git
$ cd EHR_NER/
$ conda create --name ${env_name} python=3.8
$ conda deactivate && conda activate ${env_name}
$ pip install poetry
$ poetry install
```

+ Docker

Besides, you can pull or download a docker image ([link](https://hub.docker.com/repository/docker/allenyummy/ehr_ner)) that I've pusehed to the docker hub before.
```
$ docker login
$ docker pull allenyummy/ehr_ner:0.1.1-rc
$ docker run --name ${container_name} -t -i --rm -v ${home}/EHR_NER/:/workspace ehr_ner:0.1.1-rc
```

---
## Materials for which you should prepare
+ Dataset

You should prepare your own dataet since the datset is not going to be public. However, it doesn't mean that the idea shared by the repo is not helpful. Instead, you can use QASL framework for the public nested ner corpus (ACE2004, ACE2005, GENIA, NNE).

+ Pretrained Model

There are plenty of pretrained models released by [huggingface](https://huggingface.co/models). Please download a model depending on language, model sturcture, cased/uncased, ... etc. In my case, I use `hfl/chinese-bert-wwm` as my backbone pretrained model.

+ Query

If treating NER as QASL, then it's an essential and important step to create queries. The principle to construct queris is that one Entity Type corresponds to one Query. [Li et al., (2020)](https://arxiv.org/pdf/1910.11476.pdf) conducted a series experiments on English OntoNote 5.0 with different kinds of queries (e.g., Keyword, Template, Wikipedia, Annotation Guildlines, ...). In my opinion, it's simple to use keywords as queries (i.e., the names of entity types).

---
## Data Dir Structure
```
data/
  |-- label/
  |     |-- label4qasl.txt
  |     |-- label4sl.txt
  |
  |-- query/
  |     |-- simqasl_query.json
  |     |-- qasl_query.json
  |
  |-- split/
  |     |-- train.json
  |     |-- dev.json
  |     |-- test.json

```

+ label     

label/        | label4qasl.txt | label4sl.txt
--------------|:-----:|:-----:
Format        | one line per sentence | one line per sentence
Content       | B<br>I<br>O           | B-XXX<br>I-XXX<br>B-YYY<br>I-YYY<br>...<br>O
When to use   | `make run_simqasl`    | `make run_sl`

+ query (only used for `make run_simqasl`)

query/        | simqasl_query.json | qasl_query.json
--------------|:-----:|:-----:
Format        | one question per label (json) | one question per label (json)
Content       | {"ADD": "入院日期",<br>"DCD": "出院日期",<br>...}          | {"ADD": "請找出病患的入院日期。",<br>"DCD": "請找出病患的出院日期。",<br>...}
When to use   | `make run_simqasl`    | `make run_qasl` (Now not supported yet)

+ split/train.json  (Format of dev.json or test.json is same as train.json)
```
{
    "version": "V3.0",
    "time": "2021/01/12_13:09:18",
    "data": [
        {
            "pid": 0,
            "passage": "病患於西元2019年10月5日至本院入院急診，於10月7日出院。10月16日,10月21日至本院門診追蹤治療。",
            "passage_tokens": ["病","患","於","西","元","2019","年","10","月","5","日","至","本","院","入","院","急","診",",","於","10","月","7","日","出","院","。","10","月","16","日",",","10","月","21","日","至","本","院","門","診","追","蹤","治","療","。"],
            "flat_ne_answers": [
                {
                    "type": "ADD",
                    "text": "西元2019年10月5日",
                    "start_pos": 3,
                    "end_pos": 10
                },
                {
                    "type": "DCD",
                    "text": "10月7日",
                    "start_pos": 20,
                    "end_pos": 23
                },
                {
                    "type": "OPD",
                    "text": "10月16日",
                    "start_pos": 27,
                    "end_pos": 30
                },
                {
                    "type": "OPD",
                    "text": "10月21日",
                    "start_pos": 32,
                    "end_pos": 35
                }
            ],
            "nested_ne_answers": [
                {
                    "type": "ADD",
                    "text": "西元2019年10月5日",
                    "start_pos": 3,
                    "end_pos": 10
                },
                {
                    "type": "EMD",
                    "text": "西元2019年10月5日",
                    "start_pos": 3,
                    "end_pos": 10
                },
                {
                    "type": "DCD",
                    "text": "10月7日",
                    "start_pos": 20,
                    "end_pos": 23
                },
                {
                    "type": "OPD",
                    "text": "10月16日",
                    "start_pos": 27,
                    "end_pos": 30
                },
                {
                    "type": "OPD",
                    "text": "10月21日",
                    "start_pos": 32,
                    "end_pos": 35
                }
            ]
        },
        {...},
        {...},
    ]
}
```



---
## Train on training set and Evaluate on each data set
+ Treating NER as Traditional Sequence Labeling (SL)
```
$ make run_sl
$ cat sl_config.json
{
    "data_dir": "data_dir/", ## modified it
    "train_filename": "train.json", ## modified it
    "dev_filename": "dev.json", ## modified it
    "test_filename": "test.json", ## modified it
    "labels_path": "label_dir/label_sl.txt", ## modified it
    "max_seq_length": 512,
    "overwrite_cache": false,
    "model_name_or_path": "hfl/chinese-bert-wwm",  ## download before training
    "return_dict": true,
    "with_bilstmcrf": false,  ## modified it
    "num_train_epochs": 40,
    "per_gpu_train_batch_size": 8,
    "learning_rate": 5e-5,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": false,
    "evaluate_during_training": true,
    "save_steps": 1000,
    "logging_steps": 1000,
    "eval_steps": 1000,
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": true
}
```
+ Treating NER as QA-based Sequence Labeling (QASL)
```
$ make run_simqasl
$ cat simqasl_config.json
{
    "data_dir": "data_dir",  ## modified it
    "train_filename": "train.json",
    "dev_filename": "dev.json",
    "test_filename": "test.json",
    "labels_path": "label_dir/label_simqasl.txt", ## modified it
    "query_path": "queruy_dir/query.json",  ## modified it
    "max_seq_length": 512,
    "overwrite_cache": false,
    "model_name_or_path": "hfl/chinese-bert-wwm",
    "return_dict": true,
    "with_bilstmcrf": false,  ## modified it
    "class_weights": [0.11, 1, 0.16],
    "output_dir": "exp/",
    "num_train_epochs": 40,
    "per_gpu_train_batch_size": 8,
    "learning_rate": 5e-5,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": false,
    "evaluate_during_training": true,
    "save_steps": 5000,
    "logging_steps": 1000,
    "eval_steps": 5000,
    "load_best_model_at_end": true,
    "metric_for_best_model": "eval_f1",
    "greater_is_better": true
}
```
Make sure whether the BiLSTMCRF structure is used in model or not. 

---
## Inference by a trained model
Once finishing training, you should also obtain the model performance on each data set.

If you want to check each result of each passage, you can open `api/` directory, and there are two py file that you can use: `bert_sl_predictor`, and `bert_qasl_predictor`.

```
$ PYTHONPATH=./ python bert_sl_predictor
$ PYTHONPATH=./ python bert_qasl_predictor
```

I've uploaded two models on [huggingface](https://huggingface.co/models) that have been trained on my own dataset. These two model are not equipped with BiLSTM-CRF. Instead, both models are simply BERT + Feed-Forward-Network + Softmax.

First one is based on traditional sequence labeling [allenyummy/chinese-bert-wwm-ehr-ner-sl](https://huggingface.co/allenyummy/chinese-bert-wwm-ehr-ner-sl), while second one is based on QA-based sequence labeling [allenyummy/chinese-bert-wwm-ehr-ner-qasl](https://huggingface.co/allenyummy/chinese-bert-wwm-ehr-ner-qasl).
```
from transformers import BertConfig, BertTokenizer
from models.bert_sl import BertSLModel
from models.bert_qasl import BertQASLModel

config = BertConfig.from_pretrained(self.model_dir)
tokenizer = BertTokenizer.from_pretrained(self.model_dir)
sl_model = BertSLModel.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-sl")
qasl_model = BertQASLModel.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-qasl")
```

Or you can use them as pretrained models to finetune your own downstream tasks where langugage is Chinese as well.
```
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification  

tokenizer = AutoTokenizer.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-sl")  
model = AutoModelForTokenClassification.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-sl") 

tokenizer = AutoTokenizer.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-qasl")  
model = AutoModelForTokenClassification.from_pretrained("allenyummy/chinese-bert-wwm-ehr-ner-qasl") 

```

---
## Test
Test the trained model with specific testcases.
(I'm not ready for this.)
```
$ make test_model_pred
```

---
## Citation
```
@inproceedings{chiang-etal-2021-nested,
    title = "Nested Named Entity Recognition for {C}hinese Electronic Health Records with {QA}-based Sequence Labeling",
    author = "Chiang, Yu-Lun  and
      Lin, Chih-Hao  and
      Sung, Cheng-Lung  and
      Su, Keh-Yih",
    booktitle = "Proceedings of the 33rd Conference on Computational Linguistics and Speech Processing (ROCLING 2021)",
    month = oct,
    year = "2021",
    address = "Taoyuan, Taiwan",
    publisher = "The Association for Computational Linguistics and Chinese Language Processing (ACLCLP)",
    url = "https://aclanthology.org/2021.rocling-1.3",
    pages = "18--25",
}
```
