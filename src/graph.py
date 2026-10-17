# -*- coding: utf-8 -*-
"""【第 6 步 · 图编排】用 LangGraph 把 Agent 连成「计划 → 动态路由 → 反思」的图。

一句话理解：把「计划」「统筹」「专家（带记忆+反思）」「汇总」这些节点，
用「图」连起来，规定「谁先谁后、谁决定下一步、怎么循环」。

图结构（加了「计划」「反思」和「RAG 增强」）：

    START → plan（制定计划）
              ↓
         supervisor（动态路由）
              │  next = "拉新" / "裂变_深挖" / "FINISH"
              ↓
    [专家节点：回忆记忆 → 调工具 → 推理 → 反思 → 写记忆] ──┐
       ▲                                                    │
       └────────────── 回到 supervisor ◄────────────────────┘  ← agentic 循环
    FINISH → reporter（汇总 + 反思 + RAG → 根因链 + 策略）→ END
"""
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .agents import make_agent_node
from .supervisor import make_plan_node, make_supervisor_node, make_reporter_node

STAGES = ["拉新", "激活", "留存", "裂变", "收入"]   # 5 个专家的名字清单


def _route(state: AgentState) -> str:
    """把 Supervisor 的路由决策（next）映射到「具体节点名」。"""
    nxt = state["next"]
    if nxt == "FINISH":
        return "reporter"
    return nxt.replace("_深挖", "")


def build_multi_agent(brain, memories=None, rag=None):
    """组装多智能体图。

    memories = {stage: Memory}（每个 Agent 的独立记忆，图外创建）。
    rag = RagKnowledgeBase（ChromaDB 领域知识库，供汇总 RAG 增强）。
    """
    memories = memories or {}

    g = StateGraph(AgentState)

    # —— 注册节点 ——
    g.add_node("plan", make_plan_node(brain))
    g.add_node("supervisor", make_supervisor_node(brain))
    for s in STAGES:
        g.add_node(s, make_agent_node(s, brain, memories.get(s)))
    g.add_node("reporter", make_reporter_node(brain, rag))

    # —— 连线 ——
    g.add_edge(START, "plan")
    g.add_edge("plan", "supervisor")

    # 关键① 动态路由
    g.add_conditional_edges("supervisor", _route, STAGES + ["reporter"])

    # 关键② agentic 循环：每个专家执行完都回到统筹
    for s in STAGES:
        g.add_edge(s, "supervisor")

    g.add_edge("reporter", END)

    return g.compile()


def run_multi_agent(brain, data, memories=None, rag=None, task="诊断电商增长"):
    """运行一次多智能体诊断，返回最终报告。"""
    app = build_multi_agent(brain, memories, rag)
    state = app.invoke({
        "task": task,
        "data": data,
        "messages": [],
        "tool_results": {},
        "deep_dive": [],
        "reflections": [],
        "rounds": 0,
    })
    return state["final_report"]
