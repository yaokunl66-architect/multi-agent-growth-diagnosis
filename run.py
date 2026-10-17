# -*- coding: utf-8 -*-
"""【入口】多智能体电商增长诊断系统 · 主程序。

跑法（零 Key、零网络即可）：
    python run.py

可选参数：
    python run.py --scenario gmv_decline      # 场景：GMV 下滑（默认）
    python run.py --scenario retention_crash  # 场景：留存崩盘
    python run.py --session 2026Q3-week2      # 指定会话 ID（演示跨会话记忆）

工作流（第 1~6 步）：
    1. 载入场景数据 + 组装大脑（mock / llm）
    2. 初始化团队共享 RAG 知识库（ChromaDB）
    3. 给每个专家 Agent 建独立记忆（工作记忆 + 情节记忆，情节用 ChromaDB）
    4. 用 LangGraph 组装多智能体（计划 → 动态路由 → 专家(反思) → 汇总）
    5. 运行诊断，产出根因链 + 增长策略
    6. 落盘 JSON / TXT / PNG 报告
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (LLM_MODE, LLM_CONFIG, MEMORY_DIR, RAG_DIR,
                        RAG_TOP_K, OUTPUT_DIR, SCENARIO, TASK)
from src.data_generator import generate_data
from src.rag import RagKnowledgeBase
from src.memory import Memory
from src.llm import get_brain
from src.graph import run_multi_agent
from src.report import save_report


def main():
    parser = argparse.ArgumentParser(description="多智能体电商增长诊断系统")
    parser.add_argument("--scenario", default=SCENARIO,
                        choices=["gmv_decline", "retention_crash"], help="场景名")
    parser.add_argument("--session", default=None, help="会话 ID（用于跨会话记忆演示）")
    args = parser.parse_args()

    print("=" * 60)
    print("  多智能体电商增长诊断系统（Multi-Agent + 记忆 + RAG）")
    print(f"  大脑模式：{LLM_MODE}  场景：{args.scenario}")
    print("=" * 60)

    # ① 数据 + 大脑
    data = generate_data(scenario=args.scenario)
    brain = get_brain({"LLM_MODE": LLM_MODE, "LLM_CONFIG": LLM_CONFIG})
    print(f"[1/6] 场景数据已载入：{data['scenario']['symptom']}")

    # ② 团队共享 RAG 知识库（ChromaDB）
    rag = RagKnowledgeBase(persist_dir=RAG_DIR, top_k=RAG_TOP_K)
    rag.ensure_seeded()
    print(f"[2/6] RAG 知识库就绪：{len(rag.kb)} 条领域知识（ChromaDB 向量检索 + 关键词兜底）")

    # ③ 每个 Agent 的独立记忆（工作记忆 + 情节记忆/ChromaDB，各自独立目录持久化）
    memories = {}
    for s in ["拉新", "激活", "留存", "裂变", "收入"]:
        memories[s] = Memory(agent_name=s, base_dir=os.path.join(MEMORY_DIR, s))
    print(f"[3/6] 已为 {len(memories)} 个专家 Agent 建立独立记忆（情节记忆走 ChromaDB）")

    # ④ 组装 + ⑤ 运行
    print("[4/6] 正在用 LangGraph 组装多智能体（计划 → 动态路由 → 专家反思 → 汇总）...")
    report = run_multi_agent(brain, data, memories=memories, rag=rag, task=TASK)
    print(f"[5/6] 诊断完成：共 {len(report['conversation'])} 条对话记录")

    # 沉淀记忆：本次工作记忆 → 情节记忆（跨会话可回忆）
    for m in memories.values():
        m.commit()

    # ⑥ 落盘报告
    paths = save_report(report, out_dir=OUTPUT_DIR)
    print("[6/6] 报告已生成：")
    for k, v in paths.items():
        if v:
            print(f"      {k}: {v}")

    print("=" * 60)
    print("  【最终诊断】")
    print(report["synthesis"])
    print("=" * 60)


if __name__ == "__main__":
    main()
