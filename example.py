from transformers import BertTokenizer, BertForQuestionAnswering
from models.bert_mrc import BertMRCModel 
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm')
model = BertForQuestionAnswering.from_pretrained('hfl/chinese-bert-wwm', return_dict=True)
model_2 = BertMRCModel.from_pretrained('hfl/chinese-bert-wwm')

question, text = "我是誰", "我是江侑倫"
inputs = tokenizer(question, text, return_tensors='pt')

print (inputs)

# outputs = model(**inputs)
# start_scores = outputs.start_logits
# end_scores = outputs.end_logits

# print (start_scores)
# print (end_scores)

outputs_2 = model_2(**inputs)
print (outputs_2[0])
start_logits = F.softmax(outputs_2[0], dim=2)
start_pos = torch.argmax(start_logits, dim=2)
start_pos = start_pos.detach().cpu().numpy().tolist()[0]


