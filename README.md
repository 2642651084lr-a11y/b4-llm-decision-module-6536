# B4 — Agent LLM 决策模块

> 个人模块 README — B方向 Agent 智能体


## 1. 模块概述

### 1.1 模块名称

**B4 — Agent LLM 决策模块**

### 1.2 模块说明

B4 是整个 Agent 系统的 **“大脑”和“决策引擎”** 。它负责基于对话上下文和工具描述，调用本地大语言模型（Qwen3.5-4B）进行推理，判断是否需要调用工具、调用哪个工具、传入什么参数，最终返回标准化的 AIMessage。

**本模块在系统中的作用：**

- 接收来自 B1 的对话历史（`messages`）和来自 B3 的工具描述（`tools_schema`）
- 调用本地 LLM 进行推理决策
- 返回标准 AIMessage，包含 `content`（最终回答）或 `tool_calls`（工具调用请求）

**核心设计原则：**

- **单一职责**：B4 只负责“想”和“决定”，不负责执行工具
- **松耦合**：B4 不依赖具体工具实现，只依赖工具描述（`tools_schema`）
- **可替换性**：B4 可以被替换为任何 LLM，系统其他部分无需修改

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | ✅ 全部完成 — 支持本地模型加载、Prompt 注入、JSON 输出解析、返回标准化 AIMessage |
| 进阶要求 | ✅ 任务1（多工具并发调用）已完成<br>✅ 任务3（模型动态切换）已完成<br>⚠️ 任务4（传参方式对比）代码完成，因环境问题未运行对比实验 |
| 可独立运行的演示 | ✅ 可通过命令行独立运行 `b4_local_agent_llm_clean.py` |
| 与团队系统集成情况 | ✅ 已与 B1、B3 完成联调，完整 Agent 闭环跑通 |


## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.10+ |
| 必要依赖 | transformers, torch, openai, pyyaml |
| 是否需要模型 | 需要（Qwen3.5-4B） |
| 是否需要 GPU | 推荐（H200 上运行，CPU 也可但较慢） |
| 是否需要外部数据集 | 仅测试需要（`data/messages/` 下的样例 JSON） |

### 2.2 模型依赖

| 模型 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| Qwen3.5-4B | `/root/siton-pub/production_practice/CS/assignment_B/Qwen3.5-4B` | `configs/model.yaml` 中配置 | 本地推理与工具调用决策 |

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| `messages_no_tool.json` | 项目自带 | `data/messages/` | 无工具调用的对话样例 |
| `messages_with_tool.json` | 项目自带 | `data/messages/` | 包含工具调用的对话样例 |
| `messages_multi_tool.json` | 自行构造 | `data/messages/` | 多工具并发调用测试样例 |
| `tools_schema_basic.json` | 项目自带（B3 生成） | `data/messages/` | 工具描述清单 |
| `test_suite.json` | 自行构造 | `data/` | 进阶任务4 测试用例集 |

### 2.4 安装步骤

```
# 创建并激活 conda 环境
conda create -n b4_agent_env python=3.10 -y
conda activate b4_agent_env

# 安装依赖
cd /root/sd_20236536/b4_work
pip install transformers torch openai pyyaml
```
## 3. 文件结构与接口边界

### 3.1 文件结构


```
b4_work/
├── b4_local_agent_llm_clean.py   # 主程序：B4 核心实现
├── compare_tool_calling.py       # 进阶任务4：两种传参方式对比脚本
├── debug.py                      # 调试脚本
├── configs/
│   ├── model.yaml                # 模型配置文件（路径、精度、设备等）
│   └── tools.yaml                # 工具配置文件
└── README.md                     # 本文件
```
### 3.2 接口边界


|类型|来源/去向|数据格式|说明|
|:-:|:-:|:-:|:-:|
|输入|B1 模块调用|`messages`: OpenAI 标准格式的对话历史列表|包含 system/user/assistant/tool 消息|
|输入|B3 模块生成|`tools_schema`: 工具描述 JSON 列表|每个工具含 name、description、parameters|
|输入|命令行参数|`--model_config`, `--messages`, `--tools_schema`, `--mode`|独立运行时可指定|
|输出|返回给 B1|`AIMessage`: `{"content": "...", "tool_calls": [...]}`|标准化决策结果|
|输出|保存到文件|`ai_message.json`, `raw_model_output.json`, `llm_run_log.jsonl`|调试和日志记录|
## 4. 基础要求实现与演示

### 4.1 基础功能说明

B4 模块实现了 Agent 系统的核心推理决策功能，具体包括：
1. **本地模型加载**：使用 `transformers` 加载 Qwen3.5-4B 模型，支持 `device_map` 配置（`cuda:0` / `cpu` / `auto`）
2. **Prompt 构建**：将 `tools_schema` 转换为自然语言描述，注入 system prompt，并约束模型输出纯 JSON 格式
3. **模型推理**：调用 `model.generate()`，设置确定性输出（`do_sample=False`）
4. **输出解析**：从模型原始输出中提取 JSON，解析 `content` 和 `tool_calls`
5. **标准化返回**：封装为标准 AIMessage，保存运行日志

### 4.2 基础功能实现路径


|文件/函数|作用|
|:-:|:-:|
|`b4_local_agent_llm_clean.py::generate_ai_message()`|核心入口函数，接收参数、分发模式、返回 AIMessage|
|`b4_local_agent_llm_clean.py::_load_model_bundle()`|加载本地模型和 tokenizer，带缓存机制|
|`b4_local_agent_llm_clean.py::_build_prompt_messages()`|将 `tools_schema` 转换为系统指令，约束输出格式|
|`b4_local_agent_llm_clean.py::_prompt_json_generate()`|完整推理流程：加载模型 → 构建 Prompt → 推理 → 返回原始文本|
|`b4_local_agent_llm_clean.py::_parse_model_output()`|解析模型原始输出，提取 JSON 中的 content 和 tool_calls|
**核心流程：**

```
用户问题 → B4 接收 → 加载模型 → 构建 Prompt → 模型推理 → 解析输出 → 返回 AIMessage
```
**关键代码片段（Prompt 约束）：**

```
format_instruction = (
    "IMPORTANT OUTPUT FORMAT:\n"
    "You must return exactly one valid JSON object.\n"
    "Do not output markdown.\n"
    "Do not output explanations.\n"
    "Valid schema A: {\"content\":\"final answer\",\"tool_calls\":[]}\n"
    "Valid schema B: {\"content\":\"\",\"tool_calls\":[{\"id\":\"...\",\"name\":\"...\",\"args\":{...}}]}"
)
```
### 4.3 基础功能输入格式与样例


|字段/输入文件|类型/格式|是否必需|说明|
|:-:|:-:|:-:|:-:|
|`model_config`|字符串（YAML 文件路径）|是|模型配置文件路径|
|`messages`|JSON 列表|是|OpenAI 标准对话历史|
|`tools_schema`|JSON 列表|是|B3 生成的工具描述清单|
|`mode`|字符串|否|运行模式，默认 `"prompt_json"`|
|`model_override`|字符串|否|覆盖模型路径|
**样例输入：** `data/messages/messages_no_tool.json`

```
[
  {"role": "system", "content": "You are a tool-using agent."},
  {"role": "user", "content": "请读取 docs/agent_intro.txt"}
]
```
### 4.4 基础功能演示命令


```
cd /root/sd_20236536/b4_work
conda activate b4_agent_env

python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_no_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ./outputs/demo
```
**运行后观察：**
- 模型加载成功（`model_cache=miss` 或 `model_cache=hit`）
- `ai_message.json` 在 `outputs/demo/` 目录生成
- `tool_calls` 包含正确的工具名和参数

### 4.5 基础功能输出格式


|输出文件/返回字段|格式|说明|
|:-:|:-:|:-:|
|`ai_message.json`|JSON|标准化的 AIMessage：`{"role":"assistant","content":"...","tool_calls":[...]}`|
|`raw_model_output.json`|JSON|模型的原始输出文本，用于调试|
|`llm_run_log.jsonl`|JSONL|运行日志：时间、模式、状态、文件路径|
**AIMessage 输出示例（有工具调用）：**

```
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 5000}}
    ]
}
```
### 4.6 基础功能结果截图

>运行 `python b4_local_agent_llm_clean.py` 后的终端输出和 `ai_message.json` 内容：
https://docs/images/basic_demo.png

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求


|进阶要求|是否完成|对应文件/函数|简要说明|
|:-:|:-:|:-:|:-:|
|多工具并发调用|✅ 完成|`data/messages/messages_multi_tool.json`|单次推理生成多个 `tool_calls`，实现并行工具调用|
|模型动态切换|✅ 完成|`--model` 命令行参数|通过命令行覆盖模型路径，无需修改 `model.yaml`|
|传参方式对比|⚠️ 部分完成|`_builtin_generate()`, `compare_tool_calling.py`|代码和测试集已完成，因环境问题未运行对比实验|
### 5.2 进阶功能 1：多工具并发调用

#### 功能说明

基础版本要求模型“选择恰好一个工具”。该进阶功能让模型能够在单次推理中返回多个 `tool_calls`，实现并行工具调用。
**对团队系统的帮助：** 当用户要求“同时读取两个文件”或“同时计算多个表达式”时，B4 可以一次性生成所有工具调用请求，减少多轮交互，提升系统效率。
#### 实现路径


|文件/函数|作用|
|:-:|:-:|
|`_build_prompt_messages()`|修改系统指令，从“选择一个工具”改为“可以选择零个、一个或多个工具”|
|`_parse_model_output()`|本身支持解析多个 `tool_calls`，无需改动|
|`data/messages/messages_multi_tool.json`|测试用例：要求同时读取两个文件|
**关键修改：** 系统指令从 `"Choose exactly one schema..."` 改为允许 `"one or more tool calls"`。
#### 输入格式与样例


|字段/输入文件|类型/格式|是否必需|说明|
|:-:|:-:|:-:|:-:|
|`messages_multi_tool.json`|JSON 列表|是|用户消息：`“请同时读取 agent_intro.txt 和 environment_guide.md”`|
#### 演示命令


```
python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_multi_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ./outputs/demo_multi
```
#### 输出格式


```
{
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "call_001", "name": "file_reader", "args": {"path": "docs/agent_intro.txt", "max_chars": 10000}},
        {"id": "call_002", "name": "file_reader", "args": {"path": "docs/environment_guide.md", "max_chars": 10000}}
    ]
}
```
#### 示例图片

>`ai_message.json` 中 `tool_calls` 数组包含两个元素：
https://docs/images/multi_tool_output.png

### 5.3 进阶功能 2：模型动态切换

#### 功能说明

基础版本中，切换模型需要修改 `model.yaml` 文件。该进阶功能允许通过命令行参数 `--model` 直接覆盖模型路径，无需修改配置文件。
**对团队系统的帮助：** 方便快速对比不同模型的效果，方便在模型服务不可用时切换到备用模型。
#### 实现路径


|文件/函数|作用|
|:-:|:-:|
|`build_parser()`|增加 `--model` 命令行参数|
|`_prompt_json_generate()`|接收 `model_override` 参数，覆盖 `model_config["model_name_or_path"]`|
|`generate_ai_message()`|将 `model_override` 传递给下层函数|
#### 演示命令


```
python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_no_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode prompt_json \
  --outdir ./outputs/demo_switch \
  --model /path/to/other/model
```
#### 输出格式

与基础功能相同，但模型路径被覆盖为 `--model` 指定的路径。
### 5.4 进阶功能 3：传参方式对比（代码完成，待环境验证）

#### 功能说明

对比两种告诉模型“有哪些工具可用”的方式：


|方式|实现|特点|
|:-:|:-:|:-:|
|**方式A：Prompt 注入**|工具描述写在系统指令中（已实现）|不依赖模型原生功能，适用性广|
|**方式B：Builtin**|通过 OpenAI API 的 `tools` 参数传递|原生 function calling，解析更可靠|
#### 实现路径


|文件/函数|作用|
|:-:|:-:|
|`_builtin_generate()`|使用 OpenAI SDK 调用 vLLM 服务，原生 function calling|
|`compare_tool_calling.py`|自动运行两种模式，统计成功率和耗时|
|`data/test_suite.json`|6 个测试用例覆盖不同场景|
|`--mode builtin`|命令行支持切换到 builtin 模式|
#### 演示命令（待 vLLM 服务可用）


```
python b4_local_agent_llm_clean.py \
  --model_config ./configs/model.yaml \
  --messages ./data/messages/messages_no_tool.json \
  --tools_schema ./data/messages/tools_schema_basic.json \
  --mode builtin \
  --outdir ./outputs/demo_builtin
```
#### 当前状态

- ✅ `_builtin_generate()` 函数已完整实现
- ✅ 测试集 `test_suite.json` 已设计（6 个用例）
- ✅ 对比脚本 `compare_tool_calling.py` 已编写
- ✅ 命令行 `--mode builtin` 已支持
- ❌ 因服务器 cuDNN 环境兼容性问题，vLLM 服务无法启动，对比实验未实际运行

## 6. 与团队系统的集成说明

### 6.1 调用关系

B4 模块被 **B1（Agent 运行与消息管理模块）** 调用，调用链路如下：

```
用户输入 → B1 构造 messages → B1 调用 B4.generate_ai_message() → 
B4 返回 AIMessage → B1 判断 tool_calls 是否为空 →
  若不为空：B1 调用 B3 执行工具 → B1 回填 ToolMessage → B1 再次调用 B4 →
  B4 基于工具结果生成最终答案 → B1 输出给用户
```
### 6.2 接口契约


|项目|说明|
|:-:|:-:|
|**调用方式**|B1 直接调用 `generate_ai_message()` 函数|
|**输入参数**|`messages`（对话历史）、`tools_schema`（工具清单）|
|**输出格式**|标准 AIMessage（含 `content` 或 `tool_calls`）|
|**错误处理**|解析失败时返回错误 AIMessage，不会导致系统崩溃|
### 6.3 与 B5 的关系

B4 和 B5 **没有直接交互**。B5 是记忆管理模块，它把记忆文本返回给 B1，B1 将其放入 `messages` 的 system 或 user 消息中，然后传给 B4。B4 不感知 B5 的存在，只按收到的 `messages` 做推理。
### 6.4 联调状态

- ✅ B4 生成的 `tool_calls` 能被 B1 正确捕获
- ✅ B3 能成功执行 B4 请求的工具
- ✅ 工具执行结果以标准 `ToolMessage` 回填后，B4 能继续生成最终答案
- ✅ 完整 Agent 闭环已跑通

## 7. 已知问题与后续改进


|问题|当前原因|后续改进|
|:-:|:-:|:-:|
|进阶任务4 对比实验未运行|服务器 cuDNN 环境兼容性问题，vLLM 服务无法启动|待环境修复后运行 `compare_tool_calling.py` 获取对比数据|
|模型推理速度较慢（1-2秒/次）|小模型在 H200 上推理速度有限|可尝试使用 vLLM 服务（builtin 模式）提升吞吐量|
|部分测试用例准确率不高|Qwen3.5-4B 在多步推理场景下能力有限|可升级到 Qwen3.5-7B 或更大大小的模型|
|`device` 与 `device_map` 同时存在时可能冲突|`model.to(device)` 与 `device_map` 混用|统一使用 `device` 字段控制，避免冲突|

