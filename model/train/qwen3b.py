import os, sys, pathlib, json, time, psutil
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset 

pro_path = os.path.join(pathlib.Path(__file__).parent.parent.parent)
data_path = os.path.join(pro_path, "data")
print(f"==============当前的项目路径为：{pro_path}===============")

# 模型名称
model_name = os.path.join(pro_path, "model", "model_", "qwen2.5-3b-instruct")
# model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"===============当前模型为：{model_name}=======================")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
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


model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map="auto", torch_dtype=torch.bfloat16)

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

training_arguments = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    optim = "adamw_torch",
    learning_rate = 2e-4,
    lr_scheduler_type = "cosine",
    num_train_epochs = 3,
    logging_steps = 50,
    fp16=False,
    bf16 = True,
    gradient_checkpointing=True,
    save_steps=200,
    max_steps=-1,
)

trainer = SFTTrainer(
    max_seq_length=512,
    model=model,
    args=training_arguments,
    dataset_text_field="text",
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

print("=======================开始训练=========================")
trainer.train()
print("======================训练结束===============================")

trainer.model.save_pretrained(os.path.join(output_dir, "final-result"))

# 记录实验结果

# ------- 1. 收集训练阶段汇总数据 -------
log_dict = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model_name": model_name,
    "task": "SFT",                       # 换 DPO/simPO 时改这里
    "n_train_samples": len(dataset),
    "max_seq_length": trainer.args.max_seq_length,
    "training_args": trainer.args.to_dict(),   # 所有 TrainingArguments 超参
    "peft_config": peft_config.to_dict(),      # LoRA 参数
    "final_train_loss": trainer.state.log_history[-1].get("train_loss", None),
    "total_steps": trainer.state.global_step,
    "epochs_completed": trainer.state.num_train_epochs,
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "python_version": sys.version,
    "cuda_version": torch.version.cuda,
    "transformers_version": transformers.__version__,
    "trl_version": trl.__version__,
    }

# ------- 2. 可选：在验证集上快速算 perplexity（若你传了 eval_dataset） -------
if trainer.eval_dataset is not None:
    eval_results = trainer.evaluate()
    log_dict["eval_perplexity"] = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    log_dict["eval_loss"] = eval_results["eval_loss"]

# ------- 3. 写入外部 json -------
json_path = os.path.join(output_dir, "experiment_log.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(log_dict, f, ensure_ascii=False, indent=2)

print(f"实验日志已保存至 {json_path}")

print(f"==================结果存储路径为：{output_dir}========================")