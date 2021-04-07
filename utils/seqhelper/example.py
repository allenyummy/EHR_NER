from tqdm import tqdm
from src.scheme import IOB2
from src.entity import EntityFromNestedList
input_nested_list = [[
                        ("台", "B-LOC"), 
                        ("北", "I-LOC"), 
                        ("是", "O"), 
                        ("阿", "B-PER"), 
                        ("倫", "I-PER"), 
                        ("的", "O"), 
                        ("家", "O")],
                     [
                        ("阿", "B-PER"),
                        ("倫", "I-PER"),
                        ("是", "O"),
                        ("人", "B-ANI")]
                    ]
df1 = EntityFromNestedList(input_nested_list, IOB2).chunks2df()
print (df1)
#   pid type start_position end_position text
# 0   0  LOC              0            1   台北
# 1   0  PER              3            4   阿倫
# 2   1  PER              0            1   阿倫
# 3   1  ANI              3            3    人


pred_nested_list = [[
                        ("台", "B-LOC"), 
                        ("北", "I-LOC"), 
                        ("是", "O"), 
                        ("阿", "B-PER"), 
                        ("倫", "O"), 
                        ("的", "O"), 
                        ("家", "O")],
                     [
                        ("阿", "B-PER"),
                        ("倫", "I-PER"),
                        ("是", "O"),
                        ("人", "B-ANI")]
                    ]
df2 = EntityFromNestedList(pred_nested_list, IOB2).chunks2df()
print (df2)

df1["t/p"] = "t"
df2["t/p"] = "p"
df = df1.append(df2)
df = df.sort_values(by=["pid", "start_pos"])
print (df)

# print (classification_report(input_nested_list, pred_nested_list, scheme=IOB2))
#               precision    recall  f1-score   support

#          ANI       1.00      1.00      1.00         1
#          LOC       1.00      1.00      1.00         1
#          PER       0.50      0.50      0.50         2

#    micro avg       0.75      0.75      0.75         4
#    macro avg       0.83      0.83      0.83         4
# weighted avg       0.75      0.75      0.75         4


# nested_list = list()
# temp_list = list()
# with open("train_1_update.txt", "r", encoding="utf-8") as f:
#     for line in tqdm(f):
#         line = line.rstrip()
#         if line == "":
#             nested_list.append(temp_list)
#             temp_list = list()
#         else:
#             split = line.split()
#             char = split[0]
#             label = split[1]
#             temp_list.append((char, label))

# print ("finish reading files")
# df = EntityFromNestedList(nested_list, IOB2).chunks2df()
# df.to_excel("output.xlsx", index=False)