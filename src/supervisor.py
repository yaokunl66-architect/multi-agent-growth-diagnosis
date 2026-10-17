# -*- coding: utf-8 -*-
"""【第 5 步 · 统筹】Supervisor = 制定计划 + 动态路由 + 汇总反思 + RAG 增强。

一句话理解：统筹就像「团队 leader」——
① 开工前先制定计划（哪些专家、什么顺序、依赖关系）；
② 每一步用大脑动态决定「接下来派谁干活」；
③ 大家干完后汇总结论、结合反思产出根因链 + 策略；
④ 本版新增：汇总时用 RAG（ChromaDB）检索领域知识，注入最终策略。
"""
from .knowledge import ROUTE_ORDER

MAX_STEPS = 20  # 安全上限：agentic 循环最多走 20 步，防止跑飞

# 各环节的关注点关键词——用于构建 RAG 检索 query
STAGE_QUERY = {
    "拉新": "如何降低获客成本、提升渠道 ROI",
    "激活": "如何提升新用户激活率与首单转化",
    "留存": "如何提升用户留存率、优化 Aha Moment",
    "收入": "如何提升 LTV 与复购率、放大 LTV/CAC",
    "裂变": "如何提升裂变 K 因子、设计裂变玩法",
}


def make_plan_node(brain):
    """计划节点：开工前制定显式执行计划（写进对话，全程可观测）。"""

    def node(state: dict) -> dict:
        plan = {
            "stages": ROUTE_ORDER,
            "dependency": "收入依赖拉新的 CAC，故收入放最后",
            "deep_dive": "第一轮后对 ALERT/WATCH 环节做根因深挖",
        }
        plan_text = (
            f"【统筹·计划】任务：{state.get('task', '诊断电商增长')}。"
            f"执行计划：按依赖顺序派发 {' → '.join(ROUTE_ORDER)}；"
            f"其中「{plan['dependency']}」；第一轮后对告急/关注环节做根因深挖。"
        )
        return {"messages": [plan_text], "plan": plan}

    return node


def make_supervisor_node(brain):
    """Supervisor 节点：负责「编排 + 大脑路由」。"""

    def node(state: dict) -> dict:
        deep_dive = state.get("deep_dive", [])
        if not deep_dive and all(s in state["tool_results"] for s in ROUTE_ORDER):
            deep_dive = [s for s, r in state["tool_results"].items()
                         if r.get("grade") in ("ALERT", "WATCH")]

        next_ = brain.route_next(state["messages"], state["tool_results"], deep_dive)

        rounds = state.get("rounds", 0) + 1
        if rounds > MAX_STEPS:
            next_ = "FINISH"
        return {"next": next_, "deep_dive": deep_dive, "rounds": rounds}

    return node


def make_reporter_node(brain, rag=None):
    """收官节点：汇总结论 + 反思 + RAG 知识增强，产出根因链 + 策略 + 最终报告。

    rag：RagKnowledgeBase 实例（ChromaDB）。不为 None 时，汇总会先基于
    告急/关注环节检索领域知识，注入最终策略——多智能体 + RAG 的落点。
    """

    def node(state: dict) -> dict:
        symptom = state["data"]["scenario"]["symptom"]

        # —— RAG：基于告急/关注环节，从 ChromaDB 知识库检索领域知识 ——
        knowledge, method = [], "keyword"
        if rag is not None:
            problems = [s for s, r in state["tool_results"].items()
                        if not s.endswith("_深挖") and r.get("grade") in ("ALERT", "WATCH")]
            query = "；".join(STAGE_QUERY.get(s, s) for s in problems) or "电商用户增长策略"
            knowledge, method = rag.search(query, top_k=3)

        synthesis = brain.synthesize(state["messages"], state["tool_results"],
                                     symptom=symptom, knowledge=knowledge)

        results = {k: v for k, v in state["tool_results"].items()
                   if not k.endswith("_深挖")}
        health = {
            k: {"name": k, "grade": v["grade"], "metric": v["main_metric"],
                "advice": v["advice"]}
            for k, v in results.items()
        }
        deep_dive = {k: v for k, v in state["tool_results"].items()
                     if k.endswith("_深挖")}
        reflections = list(state.get("reflections", []))

        final_report = {
            "title": "多智能体电商增长诊断报告",
            "scenario": state["data"]["scenario"],
            "task": state.get("task", ""),
            "agents": ["统筹 Agent"] + ROUTE_ORDER,
            "plan": state.get("plan", {}),
            "health": health,
            "deep_dive": deep_dive,
            "reflections": reflections,
            "knowledge": knowledge,
            "retrieval_method": method,
            "synthesis": synthesis,
            "conversation": list(state["messages"]) + [f"【统筹】{synthesis}"],
        }
        return {"messages": [f"【统筹】{synthesis}"], "final_report": final_report}

    return node
