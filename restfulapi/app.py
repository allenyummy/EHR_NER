import os
from flask import Flask, render_template
from flask import request

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification
from seqeval import get_entities

name = "allenyummy/chinese-bert-wwm-ehr-ner-sl"

config = AutoConfig.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForTokenClassification.from_pretrained(name)

def predict(text):

    tokens = tokenizer.tokenize(text)

    if len(tokens) > 510:
        tokens = tokens[:511]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    real_tokens_len = len(tokens)
    
    pad_len = 0
    if real_tokens_len < 512:
        pad_len = 512 - real_tokens_len
        tokens += ["[PAD]"] * pad_len
    
    print (tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*real_tokens_len + [0]*pad_len
    segment_ids = [0]*real_tokens_len + [0]*pad_len

    logits = model(torch.tensor([input_ids]), torch.tensor([input_mask]), torch.tensor([segment_ids]))[0]
    logits = F.softmax(logits, dim=2)
    digit_labels = torch.argmax(logits, dim=2)
    digit_labels = digit_labels.detach().cpu().numpy().tolist()[0]
    labels = [config.id2label[i] for i in digit_labels]
    
    seqs = list()
    for token, label in zip(tokens, labels):
        if token == "[PAD]":
            break
        seqs.append(label)

    entities = list()
    for ent in get_entities(seqs):
        entity = ''.join(tokens[ent[1]:ent[2]+1])
        entities.append({ent[0]: entity})

    return entities


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
      form = request.form
      result = []
      bert_abstract = form['paragraph']
      result.append(predict(bert_abstract))
      result.append(form['paragraph'])

      return render_template("index.html",result = result)

    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
