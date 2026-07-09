import json, re, torch, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/siton-pub/production_practice/CS/assignment_B/Qwen3.5-4B"
TOOL_DESC = "- calculator: Calculate arithmetic\n- file_reader: Read file"

print("Loading model once...")
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
model.eval()
print("Model loaded.\n")

def test(user_msg):
    sys = f"You are a tool assistant. Tools:\n{TOOL_DESC}\nOutput JSON with tool_calls or empty."
    msgs = [{"role":"system","content":sys},{"role":"user","content":user_msg}]
    encoded = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    # 将编码后的张量移到模型所在设备
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    out = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    return m.group(0) if m else raw

for msg in ["计算 23*17+9", "读取 docs/agent_intro.txt", "你好"]:
    try:
        raw = test(msg)
        data = json.loads(raw)
        print(f"{msg}: OK, has_tool={bool(data.get('tool_calls'))}")
    except Exception as e:
        print(f"{msg}: ERROR")
        traceback.print_exc()
        if 'raw' in locals() and raw:
            print(f"Raw output: {raw[:200]}")
