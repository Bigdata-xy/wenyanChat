# 建立一个数据处理，将train中的train.cch数据与train.mch数据合并为一个新的数据集，并保存为新的文件

import pandas as pd
import os
import pathlib

# 设置工作目录
data_path = os.path.join(pathlib.Path(__file__).parent.parent, "data")
print("数据路径: ", data_path)

# 导入数据
file_train_wenyan = os.path.join(data_path, "SCUT-C2MChn",  "train", "train.cch")
file_train_ = os.path.join(data_path, "SCUT-C2MChn",  "train", "train.mch")

# 每行去空格， 丢掉空行
with open(file_train_, encoding="utf-8") as f:
    lines_ = [l.replace(" ", "").strip() for l in f if l.strip()]

with open(file_train_wenyan, encoding="utf-8") as f:
    lines_wenyan = [l.replace(" ", "").strip() for l in f if l.strip()]

df_ = pd.DataFrame(lines_, columns=["白话文"])
df_wenyan = pd.DataFrame(lines_wenyan, columns=["文言文"])

data_combined = pd.concat([df_, df_wenyan], axis=1)
print(data_combined.info())

# 保存数据
data_combined.to_csv(os.path.join(data_path, "data_combined.csv"), index=False)
print("数据保存完成")