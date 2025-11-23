import os,pathlib, sys, time
import pandas as pd

pro_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, pro_path)

from src.llm import llm_client
from conf.model import LLM_MODEL
data_path = os.path.join(pro_path, "data")
data_combined = pd.read_csv(os.path.join(data_path, "data_combined.csv"))
data_DPO_simPO = data_combined[40000: 46000]
data_DPO_simPO_train = os.path.join(pro_path, "data", "train", "data_DPO_simPO_train_2.csv")

data_DPO_simPO["文言文_较差"] = None 

# 导入大模型
def chater(
    sys_content: str,
    user_content: str,
)-> str:
    try:
        messages = [
            {"role": "system", "content": f"{sys_content}"},
            {"role": "user", "content": f"{user_content}"}
        ]
        response = llm_client.chat(
            model = LLM_MODEL,
            messages = messages,
            stream = False,
            temperature = 0.9
        )
        return response
    except Exception as e:
        print(f"Error calling llm_client:{e}")
        return 0


for idx, row in data_DPO_simPO.iterrows():
    print(idx)
    wenyan = row["文言文"]
    sys_content = "你是一位文言文理解专家。"
    user_content = f"你是一位文言文翻译专家，给你一段正确的文言文: {wenyan}, 理解这段内容，帮我将其生成较差版的文言文。\n 注意：1. 用词粗糙或语法别扭，杂乱一些，含一半以上白话文。2. 不要有任何的解释。 3. 仅输出修改后的版本。"
    worse = chater(sys_content, user_content).choices[0].message.content
    data_DPO_simPO.at[idx, "文言文_较差"] = worse

    # 每 50 条落盘一次，断点可续
    if idx % 50 == 0:
        data_DPO_simPO.to_csv(data_DPO_simPO_train, index=False)
    time.sleep(0.2)          # 简单限速，防止接口被限