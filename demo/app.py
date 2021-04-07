# encoding = utf-8
# Author: Yu-Lun Chiang
# Description: app.py

import logging
from flask import Flask, render_template, request, url_for
from api.bert_sl_predictor import BertSLPredictor
from api.bert_qasl_predictor import BertQASLPredictor
from utils.seqhelper.src.scheme import IOB2
from utils.seqhelper.src.entity import EntityFromList, EntityElements

logger = logging.getLogger(__name__)

app = Flask(__name__)


SL_model = BertSLPredictor("allenyummy/chinese-bert-wwm-ehr-ner-sl")
sl_results = {
    "passage": "",
    "passage_tokens": "",
    "answers": [],
}

queries_map = {
    "ADD": "入院日期",
    "DCD": "出院日期",
    "SGN": "手術及處置名稱",
    "DTN": "治療及藥品名稱",
    "ICD": "入加護病房日期",
    "OCD": "出加護病房日期",
    "IBD": "入燒燙傷病房日期",
    "OBD": "出燒燙傷病房日期",
    "IND": "入負壓病房日期",
    "OND": "出負壓病房日期",
    "EMD": "急診單獨日期",
    "EMDS": "急診起始日期",
    "EMDE": "急診結束日期",
    "EMC": "急診次數",
    "OPD": "門診單獨日期",
    "OPDS": "門診起始日期",
    "OPDE": "門診結束日期",
    "OPC": "門診次數",
    "RTD": "放療單獨日期",
    "RTDS": "放療起始日期",
    "RTDE": "放療結束日期",
    "RTC": "放療次數",
    "SGD": "手術單獨日期",
    "SGDS": "手術起始日期",
    "SGDE": "手術結束日期",
    "SGC": "手術次數",
    "CTD": "化療單獨日期",
    "CTDS": "化療起始日期",
    "CTDE": "化療結束日期",
    "CTC": "化療次數",
    "DPN": "就診科別",
}
QASL_model = BertQASLPredictor(
    "allenyummy/chinese-bert-wwm-ehr-ner-qasl", queries=queries_map
)
qasl_results = {
    "passage": "",
    "passage_tokens": "",
    "answers": [],
}


def SL_predict(model, passage):
    passage_tokens = ""
    refine_answers = list()
    if passage == "":
        return passage_tokens, refine_answers
    results = SL_model.predict(passage)
    tokens, answers = SL_model.refine(results)
    passage_tokens = " / ".join(tokens)
    refine_answers = transform_query_tag_and_sort(answers)
    return passage_tokens, refine_answers


def QASL_predict(model, passage):
    passage_tokens = ""
    refine_answers = list()
    if passage == "":
        return passage_tokens, refine_answers
    tokens, answers = QASL_model.predict_overall(passage)
    passage_tokens = " / ".join(tokens)
    refine_answers = transform_query_tag_and_sort(answers)
    return passage_tokens, refine_answers


def transform_query_tag_and_sort(answers):
    refine_answers = list()
    for ans in answers:
        type = queries_map[ans.type]
        chunk = EntityElements(ans.pid, type, ans.start_pos, ans.end_pos, ans.text)
        refine_answers.append(chunk)
    return sorted(refine_answers, key=lambda x: (x.start_pos, x.end_pos, x.type))


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST" and request.values["Submit"] == "send_to_SL":
        passage = request.values["passage"]
        passage_tokens, answers = SL_predict(SL_model, passage)
        sl_results["passage"] = passage
        sl_results["passage_tokens"] = passage_tokens
        sl_results["answers"] = answers

    elif request.method == "POST" and request.values["Submit"] == "send_to_QASL":
        passage = request.values["passage"]
        qasl_results["passage"] = passage
        passage_tokens, answers = QASL_predict(QASL_model, passage)
        qasl_results["passage_tokens"] = passage_tokens
        qasl_results["answers"] = answers

    return render_template(
        "index.html", sl_results=sl_results, qasl_results=qasl_results
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)