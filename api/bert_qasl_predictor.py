# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: API of prediction from bert_qasl

import logging
import os
import json
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from models.bert_qasl import BertQASLModel

logger = logging.getLogger(__name__)


class BertQASLPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.config, self.tokenizer, self.model = self._load()
        self.id2label = self.config["id2label"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(
        self, query_tag: str = "", query: str = "", passage: str = "", top_k: int = 1
    ):
        inputs = self.tokenizer(
            query, passage, truncation=True, max_length=512, return_tensors="pt"
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
            if "##" in t:
                temp = results.pop()
                modi_t = temp[0] + t[2:]  ## 109 + ##02 -> 10902
                r = temp[1:]
                results.append((modi_t,) + r)
            else:
                r = ()
                for k in range(top_k):
                    lidp = label_id_pred[i, k]
                    lp = self.id2label[str(lidp)]
                    lp_refine = f"{lp}-{query_tag}" if lp != "O" else lp
                    p = label_id_prob[i, k]
                    r += (lp_refine, p)
                results.append((t,) + r)

        ## [CLS] Query [SEP] Passage [SEP]
        ## just keep model prediction for passage.
        first_sep_idx = tokens.index("[SEP]")
        results = results[first_sep_idx + 1 : -1]

        return results

    def _load(self):
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(f"{self.model_dir} must contain config.json.")
        config = json.load(open(config_path))
        tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        model = BertQASLModel.from_pretrained(self.model_dir, return_dict=True)
        return config, tokenizer, model


if __name__ == "__main__":

    model_dir = "trained_model/0817_8786_concat_num/qasl/2020-11-11-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-4_E-5_LR-5e-5_SD-1/"
    model = BertQASLPredictor(model_dir=model_dir)
    passage = "病患於109年10月5日入院急診。"
    passage = "病患於民國108年10月5日至本院入院急診，經手術之後，民國108年10月7日出院。"
    passage = "患者於民國109年01月20日08時20分急診就醫，經縫合手術治療後於民國109年01月20日10時50分出院。"
    passage = "病患於民國108年10月5日住院接受既定化學(Lipodox,Endoxan)治療，並於2020年05月05日出院,共住院02日。患者於2020/04/13,2020/05/04,共門診02次。"
    passage = "病患因上述原因曾於108年12月17日,108年12月26日,109年01月14日,109年02月04日,109年02月26日,109年03月24日,109年04月14日,109年05月05日,109年05月19日,109年05月28日至本院門診治療。曾於109年01月17日至109年01月23日,109年02月11日至109年02月14日,109年03月07日至109年03月10日,109年03月28日至109年03月31日,109年04月18日至109年04月21日,109年05月09日至109年05月12日住院並接受靜脈注射化學治療。(以下空白)"
    passage = "病患因上述原因,於民國109年05月18日至本院腫瘤醫學部一般病房住院,因病情需要於民國109年05月18日接受靜脈注射全身性免疫藥物與標靶藥物治療,於民國109年05月20日出院,宜於門診持續追蹤治療--以下空白--"
    passage = "病患曾於109年04月19日12:22~109年04月19日16:00至本院急診治療,於109年4月19日入院,109年4月22日行冠狀動脈繞道手術,109年4月22日至109年4月25日於加護病房治療,109年5月7日出院.出院後宜門診追蹤治療.(以下空白)"
    passage = "患者於109年05月20日至本院門診檢查接受治療至109年05月30日止,共計門診11次。(以下空白)"
    passage = "108.3.21108.4.8108.6.10108.9.9108.10.7108.12.2109.2.24109.5.18109.6.1門診"
    passage = "病患因上述病因於109年05月13日行右腕正中神經減壓手術,於109年05月14日,於109年06月01日神經外科門診追蹤治療--以下空白--"
    passage = "1,患者因前述原因,於2020-05-25至2020-05-26,共住院2日,住院接受注射標靶藥物治療。(以下空白)2,依病歷記錄,患者接受陳達人醫師於2020-05-25之本院門診追蹤治療,共計1次。(以下空白)"
    passage = "患者於109年5月27日來本院急診,自109年5月27日起至109年6月4日止來本院住院共9天,需繼續門診治療及療養。[以下空白]"
    passage = "患者於109年5月19日來本院急診。18時31分到本院急診就醫。給予腹部及胸部電腦斷層掃描,X光檢查,處置傷口及執行傷口縫合術(共1針)。宜休息3日及外科門診追蹤治療。[以下空白]"
    passage = "患者於民國109年04月04日經急診住院加護病房觀察治療後,109年04月06日病情改善,家屬要求自動出院自動出院.(以下空白)"
    passage = "住院日自108年09月09日至108年09月14日。[以下空白]"
    passage = "患者在108年12月25日07時42分至本院急診治療，經治療後，在108年12月25日住院，至108年12月27日出院。患者在108年12月30日至本院門診就醫治療。(以下空白)" 
    passage = "病患因上述原因，於2020年4月26日住院，4月27日接受人工血管置放手術，4月28日接受靜脈注射全身性化學治療，4月29日出院，宜於門診持續追蹤治療--以下空白--"

    with open("configs/qasl_query.json", "r") as f:
        qasl_query = json.load(f)

    import sys

    q_tag = sys.argv[1]
    query = qasl_query[q_tag]
    print(query)
    print(passage)
    results = model.predict(q_tag, query, passage, 3)
    import pandas as pd

    pd.set_option("display.max_rows", 200)
    df = pd.DataFrame(
        results,
        columns=[
            "Token",
            "Top1_label",
            "Top1_prob",
            "Top2_label",
            "Top2_prob",
            "Top3_label",
            "Top3_prob",
        ],
    )
    print(df)
