# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: API of prediction from bert_sl

import logging
import os
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from models.bert_sl import BertSLModel

logger = logging.getLogger(__name__)


class BertSLPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.config, self.tokenizer, self.model = self._load()
        self.id2label = self.config["id2label"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, passage: str = "", top_k: int = 1):
        inputs = self.tokenizer(
            passage, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        logits = F.softmax(logits, dim=2)
        # label_id_pred = torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()[0]
        pred = torch.topk(logits, k=top_k, dim=2)
        label_id_pred = pred.indices.detach().cpu().numpy()[0]
        label_id_prob = pred.values.detach().cpu().numpy()[0]
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs.input_ids.cpu().detach().numpy().tolist()[0]
        )

        results = list()
        for i, t in enumerate(tokens):
            r = ()
            for k in range(top_k):
                lidp = label_id_pred[i, k]
                lp = self.id2label[str(lidp)]
                p = label_id_prob[i, k]
                r += (lp, p)
            results.append((t,) + r)

        ## [CLS] Passage [SEP]
        ## just keep model prediction for passage
        results = results[1:-1]

        return results

    def _load(self):
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(f"{self.model_dir} must contain config.json.")
        config = json.load(open(config_path))
        tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        model = BertSLModel.from_pretrained(self.model_dir, return_dict=True)
        return config, tokenizer, model


if __name__ == "__main__":

    model_dir = "trained_model/0817_8786_concat_num/bert_sl/2020-09-02-00@hfl@chinese-bert-wwm@CE_S-512_B-4_E-100_LR-5e-5_SD-1"
    model = BertSLPredictor(model_dir=model_dir)
    passage = "病患於109年10月5日入院急診。"
    passage = "病患於民國108年10月5日至本院入院急診，經手術之後，民國108年10月7日出院。"
    passage2 = "患者於民國109年01月20日10時29分至急診就醫，經縫合手術治療後於民國109年01月20日10時50分離院。"

    results = model.predict(passage, 3)
    for r in results:
        print(r)
