import os
import json
import pandas as pd

pd.set_option("display.max_rows", 200)
from api.bert_qasl_predictor import BertQASLPredictor


def read_sl_data(dataset):
    sl = pd.read_table(
        os.path.join("data", "sl", "1020_23680_concat_num", dataset),
        sep=" ",
        quoting=3,
        names=["token", "ref"],
        skip_blank_lines=False,
    )
    return sl


def produce(sl, model, qasl_query):
    nan_idx_list = sl.index[sl.isnull().all(axis=1)].tolist()
    writer = pd.ExcelWriter(
        os.path.join(
            "data",
            "qasl_refined",
            "1020_23680_concat_num",
            dataset.split(".")[0] + ".xlsx",
        )
    )
    for q_tag, query in qasl_query.items():
        print(q_tag)
        results = pd.DataFrame(
            columns=[
                "token",
                "top1l",
                "top1p",
                "top2l",
                "top2p",
            ],
        )

        start = 0
        for nan_idx in nan_idx_list:
            end = nan_idx
            passage = sl["token"][start:end].values
            passage = " ".join(passage)
            r = pd.DataFrame(
                model.predict(q_tag, query, passage, top_k=2),
                columns=[
                    "token",
                    "top1l",
                    "top1p",
                    "top2l",
                    "top2p",
                ],
            )
            if len(r) < end - start:
                for i in range(end - start - len(r)):
                    r = r.append(pd.Series(), ignore_index=True)

            results = results.append(r, ignore_index=True)
            results = results.append(pd.Series(), ignore_index=True)
            start = end + 1

        concat = pd.concat([sl, results], axis=1)
        modified = concat.copy()

        final_pred = list()
        default = list()
        for i, row in concat.iterrows():
            if not pd.isna(row["ref"]):
                if row["ref"] != "O":
                    if row["top1l"] != "O":
                        final_pred.append(row["top1l"])
                        default.append(1)
                    else:
                        if row.top2p >= 0.000025:
                            final_pred.append(row["top2l"])
                            default.append(1)
                        else:
                            final_pred.append("O")
                            default.append(0)
                else:
                    final_pred.append("O")
                    default.append(0)
            else:
                final_pred.append(None)
                default.append(None)

        modified = modified.iloc[:, 0:2]
        modified["pred"] = final_pred
        modified["default"] = default

        modified.to_excel(writer, sheet_name=q_tag, index=False)
        writer.save()

    writer.save()
    writer.close()


if __name__ == "__main__":

    with open("configs/qasl_query.json", "r") as f:
        qasl_query = json.load(f)

    BertQASL_w1_model_dir = "trained_model/0817_8786_concat_num/bert_qasl/2020-11-11-00@hfl@chinese-bert-wwm@weightedCE-0.11-1-0.16_S-512_B-4_E-5_LR-5e-5_SD-1/"
    BertQASL_w1 = BertQASLPredictor(BertQASL_w1_model_dir)

    for dataset in ["dev.txt"]:
        sl = read_sl_data(dataset)
        produce(sl, BertQASL_w1, qasl_query)
