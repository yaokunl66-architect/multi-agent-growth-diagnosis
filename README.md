# 多智能体增长诊断系统（Multi-Agent）

**中文** | [English](README.en.md)

一个**零 API Key、零网络即可运行**的多智能体（Multi-Agent）电商增长诊断系统，用于教学演示与离线开发。

它把「统筹 Supervisor」和「拉新 / 激活 / 留存 / 裂变 / 收入」五个领域专家，用 **LangGraph** 编排成一张图；每个专家都有**独立记忆**（工作记忆 + 情节记忆），诊断后还会**自我反思**；统筹在汇总时通过 **RAG（ChromaDB）** 检索领域知识，注入最终策略。

> 本项目的核心能力（必选）：**ChromaDB + RAG**。
> - **团队共享 RAG 知识库**：`src/rag.py` 用 ChromaDB 建向量库，汇总时检索领域知识注入策略。
> - **情节记忆向量化**：`src/memory.py` 的情节记忆用 ChromaDB 做向量检索（不再是简单关键词匹配），每个 Agent 都有独立的向量记忆库。
> - 默认 embedding 为英文模型，中文语义召回一般，因此两处均采用 **「ChromaDB 向量检索优先 + 关键词 Jaccard 兜底」**，检索失败自动回退，保证离线可用。

---

## 目录结构

```
multi-agent-growth-diagnosis/
├── run.py                  # 入口：一键运行诊断
├── requirements.txt        # 依赖清单
├── README.md               # 本文档（中文）
├── README.en.md            # English README
└── src/
    ├── config.py           # 配置中心：大脑模式 / 记忆目录 / RAG 目录 / 场景
    ├── knowledge.py        # 专家人设、环节顺序、评级文案（第 1 步）
    ├── llm.py              # 大脑：MockBrain / LLMBrain（第 2 步）
    ├── tools.py            # 5 个精确算数工具（第 3 步）
    ├── agents.py           # 专家 Agent = 人设 + 工具 + 大脑 + 记忆 + 反思（第 4 步）
    ├── supervisor.py       # 统筹：计划 + 动态路由 + 汇总(RAG)（第 5 步）
    ├── graph.py            # LangGraph 图编排（第 6 步）
    ├── state.py            # Agent 状态定义
    ├── memory.py           # 记忆：工作记忆 + 情节记忆（ChromaDB 向量检索）
    ├── rag.py              # RAG 知识库（ChromaDB + 关键词兜底）
    ├── data_generator.py   # 多场景真实数据生成
    └── report.py           # 报告输出（JSON / TXT / PNG）
```

---

## 快速开始

```bash
# 1. 安装依赖（首次）
pip install -r requirements.txt

# 2. 运行（默认场景：GMV 下滑）
python run.py

# 3. 指定场景
python run.py --scenario retention_crash
```

运行后会在 `output/` 生成三份报告：`multi_agent_report.json` / `.txt` / `.png`。

### 首次运行需要下载 embedding 模型吗？

**不需要联网，也不会强制下载。** `src/rag.py` 和 `src/memory.py` 会先尝试 ChromaDB 默认 embedding（首次可能触发一次性本地下载）；若不可用，**自动回退到关键词 Jaccard 检索**，整个系统依然完整跑通。详见下文「RAG 与记忆」小节。

---

## 核心设计：一个「有计划的团队」

### 1. 团队分工（第 1 步）

| 角色 | 职责 | 人设示例 |
|------|------|---------|
| 统筹 Supervisor | 制定计划、动态路由、汇总反思 | 「你是统筹 Agent，负责编排」 |
| 拉新专家 | 渠道 ROI / CAC / 质量诊断 | 「你是拉新专家，专注获客效率」 |
| 激活专家 | 激活率 / 首单转化诊断 | 「你是激活专家，专注新用户转化」 |
| 留存专家 | 留存曲线 / Aha Moment 诊断 | 「你是留存专家，专注用户粘性」 |
| 裂变专家 | K 因子 / 传播诊断 | 「你是裂变专家，专注病毒传播」 |
| 收入专家 | LTV / CAC、复购诊断 | 「你是收入专家，专注商业化」 |

> 跨 Agent 依赖：收入专家需要拉新专家算出的 CAC，因此收入被排在最后。

### 2. 四个决策点（第 2 步）

大脑（`src/llm.py`）负责四处需要「思考」的地方：
1. **路由**：Supervisor 下一步派谁（`route_next`）；
2. **诊断**：专家基于工具结果 + 团队上下文 + 自己的记忆产出诊断（`diagnose`）；
3. **反思**：专家诊断完自我检查（`reflect`）；
4. **汇总**：Supervisor 汇成根因链 + 策略（`synthesize`，可注入 RAG 知识）。

两种实现：
- **MockBrain**（默认）：确定性、可复现、零 Key；
- **LLMBrain**：接 DeepSeek / OpenAI 兼容接口，改 `config.py` 的 `LLM_MODE` 即可切换。

### 3. 图编排（第 6 步）

```
START → plan → supervisor ⇄ [专家：回忆记忆→调工具→推理→反思→写记忆] → reporter → END
```

- **plan**：开工前制定显式计划（写进对话，全程可观测）；
- **supervisor**：动态路由（条件边），决定派谁或收尾；
- **专家**：每个执行完都回到 supervisor，形成 **agentic 循环**；
- **reporter**：汇总 + 反思 + RAG 增强，产出根因链 + 策略。

---

## RAG 与记忆

### 团队共享 RAG 知识库（`src/rag.py`）

- 用 **ChromaDB** 建立向量库，持久化到 `rag_db/`；
- 内置一批电商增长领域知识（如「如何降低 CAC」「提升 LTV 的 3 个抓手」等）；
- 汇总阶段，Supervisor 根据**告急/关注环节**构建查询，检索 top-K 知识注入策略；
- 返回 `(知识列表, 检索方式)`，其中 `检索方式` 为 `vector`（ChromaDB）或 `keyword`（兜底）。

### 情节记忆向量化（`src/memory.py`）

- 每个 Agent 有独立 `Memory`：
  - **工作记忆**：本轮最近若干条发现（短时）；
  - **情节记忆**：历史事件，用 **ChromaDB 向量检索**召回（不再是简单关键词匹配）；
- 持久化到 `memory/<agent>/`，跨会话可复用（演示「记性」）。

---

## 切换真实大模型（可选）

1. 安装 `langchain-openai`（见 `requirements.txt` 注释）；
2. 设置环境变量或改 `src/config.py`：

```bash
# DeepSeek
set MUTI_AGENT_LLM_MODE=deepseek
set DEEPSEEK_API_KEY=sk-xxx

# 或 OpenAI 兼容
set MUTI_AGENT_LLM_MODE=openai
set OPENAI_API_KEY=sk-xxx
set OPENAI_BASE_URL=https://api.xxx.com/v1
```

3. 重新运行 `python run.py`。

---

## 教学要点

- **为什么用图而不是硬编码流程？** 用 LangGraph 的 `add_conditional_edges` 让统筹**动态**决定下一步，而不是写死 if-else，这正是 Multi-Agent 的「智能」所在。
- **为什么要有记忆？** 没有记忆的 Agent 每次都从零开始；有了工作记忆 + 情节记忆（ChromaDB 向量检索），它才能「记得上次的发现、调用历史经验」。
- **为什么要有反思？** 诊断完先自省「结论完整吗？有没有跨环节遗漏？」，把「一次性输出」升级为「会自省的智能体」。
- **RAG 落在哪？** 汇总阶段检索领域知识注入策略，让 Multi-Agent 具备「外部知识增强」能力。

---

## 常见问题

- **报错缺 `langgraph`？** 运行 `pip install -r requirements.txt`。
- **ChromaDB 初始化慢/失败？** 系统会自动回退关键词检索，不影响整体运行。
- **想接真实大模型但没 Key？** 保持 `LLM_MODE=mock`，用 MockBrain 也能完整演示全流程。
