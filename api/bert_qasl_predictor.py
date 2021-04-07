# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: API of prediction for bert_qasl and bertbilstmcrf_qasl

import logging
import os
import json
import torch
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer
from models.bert_qasl import BertQASLModel
from models.bertbilstmcrf_qasl import BertBiLSTMCRFQASLModel
from utils.seqhelper.src.scheme import IOB2
from utils.seqhelper.src.entity import EntityFromList

logger = logging.getLogger(__name__)


class BertQASLPredictor:
    def __init__(
        self,
        model_dir: str,
        queries: dict = {},
        query_path: str = "",
        with_bilstmcrf: bool = False,
    ):
        self.model_dir = model_dir
        self.queries = queries
        self.query_path = query_path
        self.with_bilstmcrf = with_bilstmcrf
        self.class_weights = torch.FloatTensor([0.11, 1, 0.16])
        self.config, self.tokenizer, self.model = self._load()
        self.id2label = self.config.id2label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict_overall(self, passage: str = "", top_k: int = 1):
        if self.query_path != "":
            with open(self.query_path, "r", encoding="utf-8") as f:
                queries = json.load(f)
                self.queries = queries
        overall_results = list()
        for q_tag, query in self.queries.items():
            res = self.predict_for_one_query(q_tag, query, passage, top_k)
            token, ents = self.refine(res)
            overall_results.extend(ents)
        return token, overall_results

    def predict_for_one_query(
        self, query_tag: str = "", query: str = "", passage: str = "", top_k: int = 1
    ):
        # --- Preprocess input ---
        inputs = self.tokenizer(
            query, passage, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs.input_ids.cpu().detach().numpy().tolist()[0]
        )

        # --- Predict and Generate logits ---
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # --- Decode by CRF ---
        if self.with_bilstmcrf:
            label_id_pred = self.model.crf.decode(emissions=logits * self.class_weights)
            # confidence = self.model.crf(emissions=logits, tags=torch.tensor(label_id_pred))
            label_id_pred = label_id_pred[0]

        # --- Decode by Softmax and Argmax ---
        else:
            logits = F.softmax(logits, dim=2)
            # label_id_pred = torch.argmax(logits, dim=2).detach().cpu().numpy().tolist()[0]
            pred = torch.topk(logits, k=top_k, dim=2)
            label_id_pred = pred.indices.detach().cpu().numpy()[0]
            label_id_prob = pred.values.detach().cpu().numpy()[0]

        # --- Postprocess ---
        results = list()
        for i, t in enumerate(tokens):
            if "##" in t:
                temp = results.pop()
                modi_t = temp[0] + t[2:]  ## 109 + ##02 -> 10902
                r = temp[1:]
                results.append((modi_t,) + r)
            else:
                r = ()
                if self.with_bilstmcrf:
                    lidp = label_id_pred[i]
                    lp = self.id2label[lidp]
                    lp_refine = f"{lp}-{query_tag}" if lp != "O" else lp
                    r += (lp_refine,)
                else:
                    for k in range(top_k):
                        lidp = label_id_pred[i, k]
                        lp = self.id2label[lidp]
                        lp_refine = f"{lp}-{query_tag}" if lp != "O" else lp
                        p = label_id_prob[i, k]
                        r += (lp_refine, p)
                results.append((t,) + r)

        # --- Drop [CLS], Query, [SEP], [SEP]
        first_sep_idx = tokens.index("[SEP]")
        results = results[first_sep_idx + 1 : -1]
        return results

    def refine(self, results):
        # --- only support top 1 result ---
        if self.with_bilstmcrf:
            token, label = zip(*results)
        else:
            token, label, prob = zip(*results)
        seq = [(t, l) for t, l in zip(token, label)]
        ents = EntityFromList(seq=seq, scheme=IOB2).entities
        return token, ents

    def _load(self):
        config = BertConfig.from_pretrained(self.model_dir)
        tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        if self.with_bilstmcrf:
            model = BertBiLSTMCRFQASLModel.from_pretrained(
                self.model_dir, return_dict=True
            )
        else:
            model = BertQASLModel.from_pretrained(self.model_dir, return_dict=True)
        return config, tokenizer, model


if __name__ == "__main__":

    model_dir = "trained_model/0817_8786_concat_num/qasl/2020-11-11-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-4_E-5_LR-5e-5_SD-1/"
    model_dir = "trained_model/0817_8786_concat_num/simqasl/2020-12-17-07@hfl@chinese-bert-wwm@wBiLSTMCRF-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1"
    query_path = "data/final/query/simqasl_query.json"

    with_bilstmcrf = False
    if "BiLSTMCRF" in model_dir:
        with_bilstmcrf = True

    model = BertQASLPredictor(
        model_dir=model_dir, query_path=query_path, with_bilstmcrf=with_bilstmcrf
    )
    passage = "病患於民國108年10月5日至本院入院急診，經手術之後，民國108年10月7日出院。"
    passage = "患者於民國109年01月20日10時29分至急診就醫，經縫合手術治療後於民國109年01月20日10時50分離院。"
    results = model.predict_overall(passage, 1)
    for i in results:
        logger.warning(i)