import os, sys, pathlib, json, time, psutil
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
import trl
from datasets import Dataset

pro_path  = os.path.join(pathlib.Path(__file__).parent.parent.parent)
sys.path.insert(0, pro_path)

print(f"========================当前项目路径：{pro_path}============================")

data_DPO_simPO_train_path = os.path.join(pro_path,"data", "train", "data_DPO_simPO_train.csv")
data_DPO_simPO = pd.read_csv(data_DPO_simPO_train_path)
data_DPO_simPO = data_DPO_simPO.rename(columns={
    "白话文": "prompt",
    "文言文": "chosen",
    "文言文_较差": "rejected"
})

t = 5000
data_DPO_simPO_train = data_DPO_simPO.loc[:t,["prompt","chosen","rejected"]]
data_DPO_simPO_eval = data_DPO_simPO.loc[t:,["prompt","chosen","rejected"]]

dataset_train = Dataset.from_pandas(data_DPO_simPO_train)
dataset_eval = Dataset.from_pandas(data_DPO_simPO_eval)

# 模型名称
model_name = os.path.join(pro_path, "model", "model_", "qwen", "qwen", "Qwen2.5-3B-Instruct-merge-sft")
# model_name = "Qwen/Qwen2.5-3B-Instruct"
print(f"===============当前模型为：{model_name}=======================")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map="auto", torch_dtype=torch.bfloat16)

ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True)


# Lora 冻结权重，嵌套参数
peft_config = LoraConfig(
    lora_alpha = 128,
    lora_dropout = 0.05,
    r = 64,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

model = get_peft_model(model, peft_config)

# 模型训练
print("===========================模型训练阶段=========================")

output_dir = os.path.join(pro_path, "model", "results","qwen2.5-3b-instruct-dpo")

print(f"==================结果存储路径为：{output_dir}========================")

dpo_args  = DPOConfig(
    output_dir = output_dir,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    optim = "adamw_torch",
    learning_rate = 1e-4,
    warmup_ratio=0.1,
    lr_scheduler_type = "cosine",
    num_train_epochs = 1,
    logging_steps = 50,
    fp16=False,
    bf16 = True,
    gradient_checkpointing=True,
    save_total_limit=2,
    group_by_length=False,
    dataloader_drop_last=False,
    save_steps=500,
    max_steps=-1,
    remove_unused_columns=False,
    beta=0.1,
    loss_type="sigmoid"
)

#dpo_config = DPOConfig(
#    beta=0.1,          # ✅ beta 放这里
#    loss_type="sigmoid"
#)

trainer = DPOTrainer(
    model=model,
    args=dpo_args,
    ref_model=ref_model,
    train_dataset=dataset_train,
    eval_dataset=dataset_eval,
#    peft_config=peft_config,

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
    "task": "DPO",                       # 换 DPO/simPO 时改这里
    "n_train_samples": len(dataset_train),
    "max_length": trainer.args.max_length,
    "training_args": trainer.args.to_dict(),   # 所有 TrainingArguments 超参
    "peft_config": peft_config.to_dict(),      # LoRA 参数
    "final_train_loss": trainer.state.log_history[-1].get("train_loss", None),
    "total_steps": trainer.state.global_step,
    "epochs_completed": trainer.state.num_train_epochs,
    "gpu_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "python_version": sys.version,
    "cuda_version": torch.version.cuda,
    "transformers_version": "4.57.1",
    "trl_version": trl.__version__,
    }

def make_json_serializable(obj):
    """把常见非 JSON 类型转成可序列化形式"""
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, (torch.Tensor, torch.device)):
        return str(obj)          # 张量/设备直接转字符串
    return obj

# ------- 2. 可选：在验证集上快速算 perplexity（若你传了 eval_dataset） -------
if trainer.eval_dataset is not None:
    eval_results = trainer.evaluate()
    log_dict["eval_perplexity"] = torch.exp(torch.tensor(eval_results["eval_loss"])).item()
    log_dict["eval_loss"] = eval_results["eval_loss"]
else:
    log_dict["eval_loss"] = None
    log_dict["eval_perplexity"] = None

# ------- 3. 写入外部 json -------
log_dict = make_json_serializable(log_dict)
json_path = os.path.join(output_dir, "experiment_log.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(log_dict, f, ensure_ascii=False, indent=2)

print(f"实验日志已保存至 {json_path}")