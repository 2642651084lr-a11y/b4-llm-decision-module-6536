import json, re, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/root/siton-pub/production_practice/CS/assignment_B/Qwen3.5-4B"
TOOL_DESC = "- calculator: Calculate arithmetic\n- file_reader: Read file"

TESTS = [
    ("计算 23*17+9", "calculator"),
    ("读取 docs/agent_intro.txt", "file_reader"),
    ("你好", None),
]

def gen(user_msg, style):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if style == "prompt":
        sys = f"You are a tool assistant. Tools:\n{TOOL_DESC}\nOutput JSON with tool_calls or empty."
    else:
        sys = f"Assistant with functions. Tools:\n{TOOL_DESC}\nOutput JSON with tool_calls or empty."
    msgs = [{"role":"system","content":sys},{"role":"user","content":user_msg}]
    inp = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True).to(model.device)
    out = model.generate(inp, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(out[0], skip_special_tokens=True)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    return m.group(0) if m else raw

def run(style):
    ok = 0
    total = len(TESTS)
    for msg, expected in TESTS:
        raw = None
        try:
            raw = gen(msg, style)
            data = json.loads(raw)
            has = bool(data.get("tool_calls"))
            correct = (expected is None and not has) or (expected is not None and has)
            ok += 1 if correct else 0
        except Exception as e:
            print(f"\n[ERROR] {e}")
            if raw is not None:
                print(f"Raw output (first 200 chars): {raw[:200]}")
    return ok, total

print("=== prompt_injection ===")
ok, total = run("prompt")
print(f"success={ok}/{total}")

print("\n=== builtin_sim ===")
ok, total = run("builtin")
print(f"success={ok}/{total}")
