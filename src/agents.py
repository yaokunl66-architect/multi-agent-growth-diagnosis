# -*- coding: utf-8 -*-
"""【第 4 步 · 专家】领域专家 Agent = 人设 + 工具 + 大脑 + 记忆 + 反思。

一句话理解：每个专家是一个「会思考、有记性、懂自省」的智能体——
1. 有自己的人设（我是拉新专家）；
2. 会调工具算数；
3. 用大脑推理诊断；
4. 有自己的记忆（工作记忆 + 情节记忆）；
5. 诊断完会自我反思，而不是"交卷走人"。

记忆说明：每个 Agent 的 Memory 用 ChromaDB 做情节记忆的向量检索（见 memory.py）。
"""
from .knowledge import EXPERT_PERSONAS
from .tools import TOOLS

# 环节中文名 → 英文键（用于取人设/阈值/打法）
STAGE_EN = {
    "拉新": "acquisition",
    "激活": "activation",
    "留存": "retention",
    "收入": "revenue",
    "裂变": "referral",
}


def _tool_kwargs(stage: str, state: dict) -> dict:
    """给工具准备参数。关键：跨 Agent 依赖——收入专家需要拉新专家算出的 CAC。"""
    d = state["data"]
    if stage == "拉新":
        return {"channels": d["channels"]}
    if stage == "激活":
        return {"activation": d["activation"]}
    if stage == "留存":
        return {"retention": d["retention"]}
    if stage == "收入":
        cac = state["tool_results"]["拉新"]["overall_cac"]   # 跨 Agent 依赖
        return {"revenue": d["revenue"], "overall_cac": cac}
    if stage == "裂变":
        return {"referral": d["referral"]}
    raise ValueError(f"未知环节: {stage}")


def _context(state: dict, stage: str) -> str:
    """给专家的团队上下文：其他专家已经给出的评级摘要（消息通信带来的团队视野）。"""
    done = {
        k: v.get("grade")
        for k, v in state["tool_results"].items()
        if not k.endswith("_深挖") and k != stage
    }
    if not done:
        return "暂无其他专家结论"
    return "；".join(f"{k}={g}" for k, g in done.items())


def make_agent_node(stage: str, brain, memory):
    """构造一个专家 Agent 节点（memory 是它自己的独立记忆，图外创建、跨会话复用）。"""
    tool = TOOLS[stage]
    persona = EXPERT_PERSONAS[STAGE_EN[stage]]

    def node(state: dict) -> dict:
        route = state.get("next", stage)
        deep = route.endswith("_深挖")
        base = route.replace("_深挖", "")

        # ① 回忆：从「自己的记忆」取上下文（工作记忆 + 情节记忆，ChromaDB 向量检索）
        working = memory.recall_working() if memory else []
        episodic = memory.recall_episodic(stage) if memory else []

        # ② 调用领域工具（精确算数）
        result = tool.invoke(_tool_kwargs(base, state))

        # ③ 大脑推理（带团队上下文 + 自己的记忆）
        diagnosis = brain.diagnose(stage, persona, result, _context(state, base),
                                   working=working, episodic=episodic, deep=deep)

        # ④ 自我反思（决策点③）：检查结论是否完整、有无跨环节遗漏
        reflection = brain.reflect(stage, result)

        # ⑤ 写工作记忆：把本次的发现 + 反思记进自己的记忆（供下次回忆）
        if memory:
            memory.note(f"{stage}诊断：{result['main_metric']} → {result['grade']}")
            memory.note(f"{stage}反思：{reflection}")

        # ⑥ 写回团队状态：诊断写进对话、结论写进总表、反思写进反思记录
        key = base + ("_深挖" if deep else "")
        return {
            "messages": [diagnosis],
            "tool_results": {key: result},
            "reflections": [f"【{stage}·反思】{reflection}"],
        }

    return node
