# B4 - Agent LLM Decision Module

**Agent系统的"大脑"——负责推理、决策和工具调用的核心模块**

---

## 一、模块定位与核心职责

B4是整个Agent系统中唯一具备推理和决策能力的模块，承载了系统的"智能"部分。它的核心职责是：

1. **接收上下文**：接收来自B1的完整对话历史（messages）和来自B3的工具描述（tools_schema）
2. **推理决策**：调用本地大语言模型（Qwen3.5-4B），分析当前状态，判断是否需要调用工具、调用哪个工具、传入什么参数
3. **标准化输出**：将模型的原始输出解析为结构化的AIMessage，包含`content`（最终回答）和/或`tool_calls`（工具调用请求）

B4不执行工具（由B3负责），也不维护消息历史（由B1负责），遵循**单一职责原则**，确保模块可替换、可测试。

---

## 二、在系统中的位置
┌─────────────────────────────────────────────────────────────────────────────┐
│ Agent 系统架构 — B4 模块定位 │
│ │
│ ┌─────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ 用户输入 │───▶│ B1模块 │───▶│ B4模块 │───▶│ B3模块 │ │
│ │ │ │ 消息管理 │ │ ★决策引擎 │ │ 工具调度 │ │
│ └─────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│ │ B5模块 │ │ Qwen3.5 │ │ B2模块 │ │
│ │ 记忆管理 │ │ 本地模型 │ │ Skill执行 │ │
│ └─────────────┘ └─────────────┘ └─────────────┘ │
│ │
│ 核心原则：B4不直接执行工具，只负责"想"和"决定" │
│ B4不维护消息历史，只负责"基于当前上下文推理" │
│ B4可以被替换为任何LLM，系统其他部分无需修改 │
└─────────────────────────────────────────────────────────────────────────────┘

text

---

## 三、核心接口

### 函数签名

```python
def generate_ai_message(
    model_config: str,              # model.yaml 配置文件路径
    messages: list[dict],           # OpenAI标准格式的对话历史
    tools_schema: list[dict],       # B3生成的工具描述列表
    mode: str = "prompt_json",      # 运行模式
    artifact_dir: str | None = None,
    artifact_stem: str | None = None,
    model_override: str | None = None,  # 可选：覆盖模型路径
) -> dict:
输入格式
messages（OpenAI标准格式）：

json
[
    {"role": "system", "content": "You are a tool-using agent..."},
    {"role": "user", "content": "用户的问题"},
    {"role": "assistant", "content": "", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_001", "content": "工具执行结果"}
]
tools_schema（B3生成的标准格式）：

json
[
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate a safe arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "..."}
                },
                "required": ["expression"]
            }
        }
    }
]
输出格式
json
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call_001",
            "name": "file_reader",
            "args": {"path": "docs/agent_intro.txt", "max_chars": 5000}
        }
    ]
}
四、核心处理流程
text
┌──────────────────────────────────────────────────────────────┐
│                    B4 内部处理流程                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ① 读取 model.yaml 配置                                      │
│     ├── model_name_or_path                                  │
│     ├── device (cuda:0 / cpu / auto)                       │
│     ├── torch_dtype (bfloat16 / float16)                   │
│     └── generation (max_new_tokens, temperature, top_p)    │
│                                                              │
│  ② 加载本地模型                                              │
│     ├── AutoTokenizer.from_pretrained()                    │
│     └── AutoModelForCausalLM.from_pretrained()             │
│                                                              │
│  ③ 构建 Prompt（关键步骤）                                   │
│     ├── 将 tools_schema 转换为文本描述                       │
│     ├── 将工具描述注入 system prompt                        │
│     ├── 约束输出格式为 JSON                                 │
│     └── 合并 messages 构造完整输入                          │
│                                                              │
│  ④ 调用模型推理                                              │
│     └── model.generate(inputs, max_new_tokens, ...)        │
│                                                              │
│  ⑤ 解析模型输出                                              │
│     ├── 正则提取 JSON                                       │
│     ├── 解析 content 和 tool_calls                         │
│     └── 验证格式合法性                                      │
│                                                              │
│  ⑥ 返回标准化 AIMessage                                     │
│     ├── 保存 raw_model_output.json（原始输出）              │
│     ├── 保存 ai_message.json（解析后的AIMessage）           │
│     └── 保存 llm_run_log.jsonl（运行日志）                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
五、支持的模式
模式	后端	工具传递方式	适用场景
prompt_json	Transformers	工具描述注入System Prompt	默认模式，无需外部服务
builtin	vLLM (OpenAI API)	原生Function Calling	需要vLLM服务的高吞吐场景
builtin_sim	Transformers	模拟Function Calling风格	对比实验，无需额外服务
六、关键文件说明
文件	说明
b4_local_agent_llm_clean.py	主程序，包含所有核心功能
configs/model.yaml	模型配置文件
compare_tool_calling.py	两种传参方式对比脚本
debug.py	调试脚本，用于快速验证
七、命令行用法
基础运行
bash
python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_no_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ./outputs/demo
模型动态切换（进阶任务3）
bash
python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_no_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ./outputs/demo \
  --model /path/to/other/model
查看输出
bash
cat ./outputs/demo/ai_message.json | python -m json.tool
八、已完成进阶任务
✅ 任务1：多工具并发调用
支持单次AIMessage生成多个tool_calls，实现并行工具调用。

测试用例：data/messages/messages_multi_tool.json

验证结果：

json
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 10000}},
        {"id": "call_002", "name": "file_reader", "args": {"path": "docs/environment_guide.md", "max_chars": 10000}}
    ]
}
✅ 任务3：模型动态切换
支持通过--model命令行参数覆盖模型路径，无需修改model.yaml文件。

九、联调状态
✅ 已与B1、B3完成完整联调

联调验证了以下闭环路径：

text
用户输入 → B1构造messages → B4生成tool_calls → B3调度执行工具 → 
ToolMessage回填B1 → B4二次推理 → 最终答案
B3同学确认：B4生成的tool_calls能被正确捕获并由B3成功调度执行，执行结果以标准ToolMessage回填后，B4能继续生成最终答案。

十、环境要求
Python 3.10+

transformers

torch (CUDA版本)

vllm（可选，用于builtin模式）

十一、后续改进方向
支持Plan-and-Execute架构

集成更多开源模型进行对比实验

增加工具调用结果缓存

十二、参考资料
ReAct: Synergizing Reasoning and Acting in Language Models

Qwen3.5-4B Model Card

vLLM Documentation
