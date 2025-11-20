import os, sys, pathlib
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset


pro_path = os.path.join(pathlib.Path(__file__).parent.parent.parent)
data_path = os.path.join(pro_path, "data")
print(f"==============当前的项目路径为：{pro_path}===============")

# 模型名称
model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"===============当前模型为：{model_name}=======================")

# # 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

# 数据导入
test_dataset = pd.read_csv(os.path.join(data_path, "data_combined.csv"))
t = 5000
print(f"===================当前测试数据长度为：{t}===========================")

def format_prompt(example):
    chat = [
        {"role": "system","content": "你是一个非常棒的人工智能助手。"},
        {"role": "user", "content": example["白话文"]},
        {"role": "assistant", "content": example["文言文"]}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}
    
raw_df = test_dataset[:t]                                    # 1. 取前 5000 行（仍是 pandas）
dataset = Dataset.from_pandas(raw_df)                        # 2. 转成 datasets.Dataset
dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)  # 3. 再 map

print("=====================训练数据示例为：====================")
print(dataset[0])


model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)

# Lora 冻结权重，嵌套参数
peft_config = LoraConfig(
    lora_alpha = 32,
    lora_dropout = 0.1,
    r = 64,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ['k_proj', 'v_proj', 'q_proj']
)

model = get_peft_model(model, peft_config)

# 模型训练
print("===========================模型训练阶段=========================")

output_dir = os.path.join(pro_path, "model", "results", model_name)

print(f"==================结果存储路径为：{output_dir}========================")

