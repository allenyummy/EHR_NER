import os
import json
import pandas as pd
from tqdm import tqdm
from src.scheme import IOB2
from src.entity import EntityFromList
from api.bert_sl_predictor import BertSLPredictor
from api.bert_qasl_predictor import BertQASLPredictor

df = pd.DataFrame(
    columns=[
        "passage",
        "flat_ne_answers",
        "nested_ne_answers",
        "bert_sl_pred",
        "bertbilstmcrf_sl_pred",
        "bert_qasl_pred",
        "bertbilstmcrf_qasl_pred",
    ]
)

data_path = os.path.join(
    "data", "final", "V3.0", "nestedner_dev.json"
)
query_path = os.path.join(
    "data", "final", "query", "simqasl_query.json"
)
out_data_path = os.path.join("results", "overall_dev_checked.xlsx")

with open(query_path, "r", encoding="utf-8") as fq:
    query = json.load(fq)

with open(data_path, "r", encoding="utf-8") as f:
    in_data = json.load(f)
data = in_data["data"]

Bert_SL_model = BertSLPredictor(
    model_dir="trained_model/0817_8786_concat_num/sl/2020-12-10-05@hfl@chinese-bert-wwm@CE_S-512_B-8_E-20_LR-5e-5_SD-1",
    with_bilstmcrf=False,
)
BertBiLSTMCRF_SL_model = BertSLPredictor(
    model_dir="trained_model/0817_8786_concat_num/sl/2020-12-14-09@hfl@chinese-bert-wwm@BiLSTMCRF_S-512_B-8_E-20_LR-5e-5_SD-1",
    with_bilstmcrf=True,
)
Bert_QASL_model = BertQASLPredictor(
    model_dir="trained_model/0817_8786_concat_num/simqasl/2020-12-10-07@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1",
    with_bilstmcrf=False,
)
BertBiLSTMCRF_QASL_model = BertQASLPredictor(
    model_dir="trained_model/0817_8786_concat_num/simqasl/2020-12-17-07@hfl@chinese-bert-wwm@wBiLSTMCRF-0.11-1-0.16_S-512_B-8_E-20_LR-5e-5_SD-1",
    with_bilstmcrf=True,
)


for i, d in enumerate(tqdm(data)):

    pid = d["pid"]
    passage = d["passage"]
    passage_tokens = d["passage_tokens"]
    input_passage = " ".join(passage_tokens)
    flat_ne_answers = d["flat_ne_answers"]
    nested_ne_answers = d["nested_ne_answers"]

    # --- Generate flat ne ansewers ---
    flat_ne_answers_results = list()
    flat_ne_answers_set = set()
    for r in flat_ne_answers:
        if r["type"] == "DIN":   ## QASL doesn't handle this label
            pass
        else:
            o = f"[{r['type']}][{r['text']}][{r['start_pos']}-{r['end_pos']}]"
            flat_ne_answers_results.append(o)
            flat_ne_answers_set.add((r["type"], r["text"], r["start_pos"], r["end_pos"]))
    flat_ne_answers_results = ",\n".join(flat_ne_answers_results)
            
    # --- Generate nested ne ansewers ---
    nested_ne_answers_results = list()
    nested_ne_answers_set = set()
    for r in nested_ne_answers:
        o = f"[{r['type']}][{r['text']}][{r['start_pos']}-{r['end_pos']}]"
        nested_ne_answers_results.append(o)
        nested_ne_answers_set.add((r["type"], r["text"], r["start_pos"], r["end_pos"]))
    nested_ne_answers_results = ",\n".join(nested_ne_answers_results)

    # --- Predict by Bert_SL_model ---
    res = Bert_SL_model.predict(input_passage)
    token, label, prob = zip(*res)
    seq = [(t, l) for t, l in zip(token, label)]
    bert_sl_pred_entities = EntityFromList(seq=seq, scheme=IOB2).entities
    bert_sl_pred_results = list()
    bert_sl_pred_entities_set = set()
    for r in bert_sl_pred_entities:
        o = f"[{r.type}][{r.text}][{r.start_pos}-{r.end_pos}]"
        bert_sl_pred_results.append(o)
        bert_sl_pred_entities_set.add((r.type, r.text, r.start_pos, r.end_pos))
    bert_sl_pred_results = ",\n".join(bert_sl_pred_results)
    

    # --- Predict by BertBiLSTMCRF_SL_model ---
    res = BertBiLSTMCRF_SL_model.predict(input_passage)
    token, label = zip(*res)
    seq = [(t, l) for t, l in zip(token, label)]
    bertbilstmcrf_sl_pred_entities = EntityFromList(seq=seq, scheme=IOB2).entities
    bertbilstmcrf_sl_pred_results = list()
    bertbilstmcrf_sl_pred_entities_set = set()
    for r in bertbilstmcrf_sl_pred_entities:
        o = f"[{r.type}][{r.text}][{r.start_pos}-{r.end_pos}]"
        bertbilstmcrf_sl_pred_results.append(o)
        bertbilstmcrf_sl_pred_entities_set.add((r.type, r.text, r.start_pos, r.end_pos))
    bertbilstmcrf_sl_pred_results = ",\n".join(bertbilstmcrf_sl_pred_results)

    # --- Predict by Bert_QASL_model ---
    bert_qasl_pred_entities = list()
    for t, q in query.items():
        res = Bert_QASL_model.predict(t, q, input_passage)
        token, label, prob = zip(*res)
        seq = [(t, l) for t, l in zip(token, label)]
        bert_qasl_pred_entities.extend(EntityFromList(seq=seq, scheme=IOB2).entities)
    bert_qasl_pred_results = list()
    bert_qasl_pred_entities_set = set()
    for r in sorted(bert_qasl_pred_entities, key=lambda x: x.start_pos):
        o = f"[{r.type}][{r.text}][{r.start_pos}-{r.end_pos}]"
        bert_qasl_pred_results.append(o)
        bert_qasl_pred_entities_set.add((r.type, r.text, r.start_pos, r.end_pos))
    bert_qasl_pred_results = ",\n".join(bert_qasl_pred_results)

    # --- Predict by BertBiLSTMCRF_QASL_model ---
    bertbilstmcrf_qasl_pred_entities = list()
    for t, q in query.items():
        res = BertBiLSTMCRF_QASL_model.predict(t, q, input_passage)
        token, label = zip(*res)
        seq = [(t, l) for t, l in zip(token, label)]
        bertbilstmcrf_qasl_pred_entities.extend(EntityFromList(seq=seq, scheme=IOB2).entities)
    bertbilstmcrf_qasl_pred_results = list()
    bertbilstmcrf_qasl_pred_entities_set = set()
    for r in sorted(bertbilstmcrf_qasl_pred_entities, key=lambda x: x.start_pos):
        o = f"[{r.type}][{r.text}][{r.start_pos}-{r.end_pos}]"
        bertbilstmcrf_qasl_pred_results.append(o)
        bertbilstmcrf_qasl_pred_entities_set.add((r.type, r.text, r.start_pos, r.end_pos))
    bertbilstmcrf_qasl_pred_results = ",\n".join(bertbilstmcrf_qasl_pred_results)

    # --- Overall Result ---
    df.loc[pid, "passage"] = passage
    df.loc[pid, "flat_ne_answers"] = flat_ne_answers_results
    df.loc[pid, "nested_ne_answers"] = nested_ne_answers_results
    df.loc[pid, "bert_sl_pred"] = bert_sl_pred_results
    df.loc[pid, "bertbilstmcrf_sl_pred"] = bertbilstmcrf_sl_pred_results
    # df.loc[pid, "bert_qasl_pred"] = bert_qasl_pred_results
    # df.loc[pid, "bertbilstmcrf_qasl_pred"] = bertbilstmcrf_qasl_pred_results

    df.loc[pid, "flatne_tn"] = len(flat_ne_answers_set)
    df.loc[pid, "nestedne_tn"] = len(nested_ne_answers_set)

    df.loc[pid, "bert_sl_pn"] = len(bert_sl_pred_entities_set)
    df.loc[pid, "bert_sl_fcn"] = len(flat_ne_answers_set & bert_sl_pred_entities_set)
    df.loc[pid, "bert_sl_ncn"] = len(nested_ne_answers_set & bert_sl_pred_entities_set)

    df.loc[pid, "bertbilstmcrf_sl_pn"] = len(bertbilstmcrf_sl_pred_entities_set)
    df.loc[pid, "bertbilstmcrf_sl_fcn"] = len(flat_ne_answers_set & bertbilstmcrf_sl_pred_entities_set)
    df.loc[pid, "bertbilstmcrf_sl_ncn"] = len(nested_ne_answers_set & bertbilstmcrf_sl_pred_entities_set)

    df.loc[pid, "bert_qasl_pn"] = len(bert_qasl_pred_entities_set)
    df.loc[pid, "bert_qasl_fcn"] = len(flat_ne_answers_set & bert_qasl_pred_entities_set)
    df.loc[pid, "bert_qasl_ncn"] = len(nested_ne_answers_set & bert_qasl_pred_entities_set)

    df.loc[pid, "bertbilstmcrf_qasl_pn"] = len(bertbilstmcrf_qasl_pred_entities_set)
    df.loc[pid, "bertbilstmcrf_qasl_fcn"] = len(flat_ne_answers_set & bertbilstmcrf_qasl_pred_entities_set)
    df.loc[pid, "bertbilstmcrf_qasl_ncn"] = len(nested_ne_answers_set & bertbilstmcrf_qasl_pred_entities_set)

#--- overall
flatne_tn_all = df["flatne_tn"].sum()
nestedne_tn_all = df["nestedne_tn"].sum()

bert_sl_pn_all = df["bert_sl_pn"].sum()
bert_sl_fcn_all = df["bert_sl_fcn"].sum()
bert_sl_ncn_all = df["bert_sl_ncn"].sum()

bertbilstmcrf_sl_pn_all = df["bertbilstmcrf_sl_pn"].sum()
bertbilstmcrf_sl_fcn_all = df["bertbilstmcrf_sl_fcn"].sum()
bertbilstmcrf_sl_ncn_all = df["bertbilstmcrf_sl_ncn"].sum()

bert_qasl_pn_all = df["bert_qasl_pn"].sum()
bert_qasl_fcn_all = df["bert_qasl_fcn"].sum()
bert_qasl_ncn_all = df["bert_qasl_ncn"].sum()

bertbilstmcrf_qasl_pn_all = df["bertbilstmcrf_qasl_pn"].sum()
bertbilstmcrf_qasl_fcn_all = df["bertbilstmcrf_qasl_fcn"].sum()
bertbilstmcrf_qasl_ncn_all = df["bertbilstmcrf_qasl_ncn"].sum()

for i, (pn, fcn, ncn) in enumerate(zip(
    [bert_sl_pn_all, bertbilstmcrf_sl_pn_all, bert_qasl_pn_all, bertbilstmcrf_qasl_pn_all],
    [bert_sl_fcn_all, bertbilstmcrf_sl_fcn_all, bert_qasl_fcn_all, bertbilstmcrf_qasl_fcn_all],
    [bert_sl_ncn_all, bertbilstmcrf_sl_ncn_all, bert_qasl_ncn_all, bertbilstmcrf_qasl_ncn_all]
)):
    print (f"************* {i} **************")
    print ("===== FLAT NE RESULTS =====")
    flat_p = fcn/pn
    flat_r = fcn/flatne_tn_all
    flat_f = 2 * flat_p * flat_r / (flat_p + flat_r)
    print (f"p: {flat_p}")
    print (f"r: {flat_r}")
    print (f"f: {flat_f}")

    print ("===== NESTED NE RESULTS =====")
    nest_p = ncn/pn
    nest_r = ncn/nestedne_tn_all
    nest_f = 2 * nest_p * nest_r / (nest_p + nest_r)
    print (f"p: {nest_p}")
    print (f"r: {nest_r}")
    print (f"f: {nest_f}")
    print ()

df.to_excel(out_data_path, index=False)
