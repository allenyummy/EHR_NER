# Named Entity Recognition of Electrical Health Record

## Sequence Labeling

---
## Machine Reading Comprehension

---
## Virtual Environment
+ Local virtual conda environment
```
$ git clone https://github.com/allenyummy/EHR_NER.git
$ cd EHR_NER/
$ conda create --name ${env_name} python=3.6.9
$ conda deactivate && conda activate ${env_name}
$ pip install poetry
$ poetry install
```
+ Docker
```
$ docker pull allenyummy/ehr_ner:0.1.0
$ docker run --name ${container_name} -t -i --rm -v ${home}/EHR_NER/:/workspace ehr_ner:0.1.0
```

---

## Citation
```
@article{Comimg soon,
  author={Yu-Lun Chiang, Giles ChaoGG},
  mail={chiangyulun0914@gmail.com}
}
```
