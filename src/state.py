# -*- coding: utf-8 -*-
"""【第 1 步 · 状态】多智能体共享的「黑板」+ 每个 Agent 的独立记忆。

一句话理解：这里有两层状态——
1. 团队共享状态（AgentState）：所有 Agent 都能读写的「公共黑板」，
   用于消息通信（对话）和结论共享（tool_results）；
2. 每个 Agent 的独立记忆（Memory，见 memory.py）：每个 sub-agent 的「私有状态」，
   团队看不到，只有自己能用（工作记忆 + 情节记忆）。

为什么需要区分「共享」和「私有」？—— 真·多智能体里，Agent 之间
既要有「公开对话」（协作），又要有「私下笔记」（各自思考），二者不能混在一起。
"""
from typing import Annotated, TypedDict
import operator


def merge_dicts(left: dict, right: dict) -> dict:
    """把两个字典合并成一个（后面的覆盖前面的同名 key）。

    为什么要自己写？—— LangGraph 里，多个专家节点会各自返回一部分结论
    （如拉新返回 {"拉新": ...}、收入返回 {"收入": ...}）。
    如果不用合并函数，后写的会「整个覆盖」先写的，结论就丢了。
    """
    return {**left, **right}


class AgentState(TypedDict, total=False):
    """团队共享状态（公共黑板）。每个字段的含义如下。"""

    # 任务描述（本次要诊断什么问题，如「GMV 环比下滑 12%，定位根因」）
    task: str

    # 原始业务数据（AARRR 五环节数据），专家们从这里取数
    data: dict

    # 团队对话历史：每个 Agent 说一句话就追加一条（operator.add = 按顺序拼接）
    messages: Annotated[list, operator.add]

    # 各专家的结构化结论（merge_dicts 保证多个专家写的结论拼成一张总表）
    tool_results: Annotated[dict, merge_dicts]

    # Supervisor 的路由决策：下一步派谁（"拉新" / "裂变_深挖" / "FINISH"）
    next: str

    # 已经走了多少步（防止 agentic 循环死循环）
    rounds: int

    # 需要「深挖」的环节清单（第一轮发现告急后，第二轮针对它们追根因）
    deep_dive: list

    # 各专家的「自我反思」记录（operator.add 追加，团队可见——反思也是协作的一部分）
    reflections: Annotated[list, operator.add]

    # 最终报告（汇总节点产出）
    final_report: dict
