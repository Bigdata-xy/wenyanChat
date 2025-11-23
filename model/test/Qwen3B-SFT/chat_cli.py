from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset

pro_path = os.path.join(pathlib.Path(__file__).parent.parent.parent.parent)

BASE_MODEL_DIR = os.path.join(pro_path, "model", "model_", "qwen", "qwen", "Qwen2.5-3B-Instruct") 
ADAPTER_DIR = os.path.join(pro_path, "model", "results", "qwen2.5-3b-instruct-sft", "final-result") 

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, local_files_only=True)

print("Loading base model …")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 4. 挂载 LoRA
print("Loading LoRA …")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

# 5. 流式输出（可选，不想要可注释掉 streamer 一行）
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

print("模型加载完成！输入问题即可（exit 退出）\n")

# 6. 命令行交互
history = []  # 多轮记忆
while True:
    try:
        user = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if user.lower() in {"exit", "quit"}:
        break

    history.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer       # 如需流式打印
        )

    if not isinstance(streamer, TextStreamer):  # 非流式时手动解码
        response = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(response.strip())

    # 把助手回答也写进历史，方便多轮
    assistant_text = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": assistant_text.strip()})