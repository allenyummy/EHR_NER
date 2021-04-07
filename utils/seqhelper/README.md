# Seqhelper

Seqhelper is a Python framework for sequence labeling, such as Named Entity Recognition, Part of Speech tagging, and so on.

The origin idea is derived from [chakki-works/seqeval](https://github.com/chakki-works/seqeval), but not just for evaluation.
I add the machanism to check if tag is available. Also, you can use entity.py to generate dataframe for entities.

Now the repo only supports IOB2 scheme. Other scheme will be available in the future.

## Get Entities
Get Entities From Nested List
```
from src.scheme import IOB2
from src.entity import EntityFromNestedList
worker = EntityFromNestedList(input_nested_list, IOB2)
entities = worker.entities ## Tuple[[(pid, type, start_position, end_position, text), ...]]
df = worker.chunks2df() ## Pandas.DataFrame
```
Get Entities From List
```
from src.scheme import IOB2
from src.entity import EntityFromList
worker = EntityFromList(input_list, IOB2)
entities = worker.entities
```
## Evaluation
Evaluate for overall performance report
```
from src.scheme import IOB2
from src.eval import f1_score, precision_score, recall_score
f1 = f1_score(trues, preds, IOB2)
p = precision_score(trues, preds, IOB2)
r = recall_score(trues, preds, IOB2)
```
Evaluate for classification report
```
from src.chakki_works.seqeval.v1 import classification_report
print (classification_report(input_nested_list, pred_nested_list, scheme=IOB2))
```
## Test
```
PYTHONPATH=./ pytest --log-cli-level=warning --cov=./  
```
## Citation
```
@{seqhelper,
  title={{seqhelper}: A Python framework for sequence labeling},
  url={https://github.com/allenyummy/seqhelper},
  note={Software available from https://github.com/allenyummy/seqhelper},
  author={Yu-Lun Chiang},
  year={2020},
}
```