import pandas as pd, pathlib, os, sys

# 设置工作目录
data_path = os.path.join(pathlib.Path(__file__).parent.parent, "data")
print("数据路径: ", data_path)


# 导入数据
data_taobao = pd.read_excel(os.path.join(data_path, "文言文&白话文互译数据集-17087条.xlsx"), names=["白话文", "文言文"])

data_taobao["文言文"] = data_taobao["文言文"].str.lstrip(":：")

data_taobao["文言文"] = data_taobao["文言文"].str.split(r"\n|:|：", n=1).str[0]

print(f"处理解释后的前五行数据: {data_taobao.head()}")

# 保存数据
data_taobao.to_excel(os.path.join(data_path, "data_taobao_processed.xlsx"), index=False)
print("数据保存完成")
