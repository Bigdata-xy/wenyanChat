# csv2pref.py
import os,pathlib, sys, time, uuid,json
import pandas as pd

pro_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, pro_path)


csv_file = os.path.join(pro_path,"data", "train", "data_DPO_simPO_train_2.csv")
csv_output = os.path.join(pro_path,"data", "train", "data_simPO_train.jsonl")

df = pd.read_csv(csv_file)
df = df.dropna()                 # 先删 NaN
df = df[(df != "").all(axis=1)]  # 再删空字符串

with open(csv_output, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        sample = {
        "id": str(uuid.uuid4()),
        "conversations": [
        {"from": "human", "value": f"把下面句子翻译成文言文：{row['白话文'].strip()}"}
        ],
        "chosen": {"from": "gpt", "value": row["文言文"].strip()},
        "rejected": {"from": "gpt", "value": row["文言文_较差"].strip()}
        }
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"✅ 已生成 {len(df)} 条 preference 样本 -> {csv_output}")