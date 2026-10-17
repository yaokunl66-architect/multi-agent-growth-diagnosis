# -*- coding: utf-8 -*-
"""【第 2 步 · 大脑】每个 Agent 用来推理的「思维中枢」。

一句话理解：大脑 = Agent 的「脑」，负责四件需要「思考」的事——
① 下一步派谁（路由）② 这个环节怎么诊断 ③ 诊断完自我反思 ④ 汇总成根因链。
工具负责「算数」，大脑负责「判断」——这就是多智能体的智能所在。

本版新增「反思」决策点：每个 Agent 诊断后不直接交卷，而是先自我检查
「我的结论完整吗？有没有漏掉跨环节影响？置信度合理吗？」，反思结果
写进记忆 + 团队对话。这让 Agent 从「一次性输出」进化为「会自省的智能体」。

为什么要做成「可插拔」？—— 离线开发/教学用确定性 MockBrain（零 Key、零网络），
生产环境接真实大模型 LLMBrain（DeepSeek/OpenAI）。改一行配置就能切换。
"""
import os

from .knowledge import ROUTE_ORDER


class Brain:
    """大脑基类：定义多智能体的「四个决策点」。子类（Mock/LLM）各自实现。"""

    def route_next(self, messages, tool_results, deep_dive) -> str:
        """决策点①：Supervisor 路由——返回下一步派发的 Agent 名 / "FINISH"。"""
        raise NotImplementedError

    def diagnose(self, agent_name, persona, tool_result, context,
                 working=None, episodic=None, deep=False) -> str:
        """决策点②：专家诊断——基于工具结果 + 上下文 + 自己的记忆，产出诊断。"""
        raise NotImplementedError

    def reflect(self, agent_name, tool_result) -> str:
        """决策点③：专家自我反思——诊断完检查结论是否完整、有无遗漏。"""
        raise NotImplementedError

    def synthesize(self, messages, tool_results, symptom="增长指标下滑",
                   knowledge=None) -> str:
        """决策点④：Supervisor 汇总——产出根因链 + 综合策略。

        knowledge：RAG 从 ChromaDB 知识库检索到的领域知识（可选注入）。
        """
        raise NotImplementedError


class MockBrain(Brain):
    """确定性大脑：零 Key、零网络、可复现，用于离线测试与教学演示。"""

    def route_next(self, messages, tool_results, deep_dive):
        """路由逻辑（确定性版）：下一步派谁。"""
        done = set(tool_results.keys())
        pending = [s for s in deep_dive if f"{s}_深挖" not in done]
        if pending:
            return f"{pending[0]}_深挖"
        for stage in ROUTE_ORDER:
            if stage not in done:
                return stage
        return "FINISH"

    def diagnose(self, agent_name, persona, tool_result, context,
                 working=None, episodic=None, deep=False):
        """诊断逻辑（确定性版）：把工具结果 + 记忆拼成结构化诊断。"""
        grade = tool_result.get("grade", "OK")
        verdict = {"ALERT": "告急", "WATCH": "需关注", "OK": "健康"}[grade]
        main = tool_result.get("main_metric", "")
        extra = tool_result.get("extra", "")
        context_note = f"（结合上下文：{context}）" if context else ""

        memory_note = ""
        if episodic:
            memory_note = f"（回忆历史经验：{'；'.join(episodic[-2:])}）"

        if deep:
            root = tool_result.get("root_cause", "根因待进一步定位")
            return (f"【{agent_name}】深挖根因：{main}评级「{verdict}」，"
                    f"根因是 {root}。{extra}{context_note}{memory_note}")
        return (f"【{agent_name}】诊断结论：{main}，评级「{verdict}」。"
                f"{extra}{context_note}{memory_note}")

    def reflect(self, agent_name, tool_result) -> str:
        """自我反思（确定性版）：检查结论完整性与跨环节影响。"""
        grade = tool_result.get("grade", "OK")
        if grade in ("ALERT", "WATCH"):
            return (f"{agent_name}自我反思：本环节{grade}，"
                    f"结论已给出根因与建议，但需确认是否受上游环节传导影响，建议深挖。")
        return (f"{agent_name}自我反思：本环节健康，"
                f"但需警惕被上游隐患波及的风险，已记录待观察。")

    def synthesize(self, messages, tool_results, symptom="增长指标下滑",
                   knowledge=None):
        """汇总逻辑（确定性版）：产出「数据驱动的因果链」+ 分级策略。

        本版新增：把 RAG 检索到的领域知识（knowledge）注入最终策略。
        """
        from .knowledge import FUNNEL_ORDER

        base = {k: v for k, v in tool_results.items() if not k.endswith("_深挖")}
        severity = {"ALERT": 0, "WATCH": 1, "OK": 2}

        problems = [s for s in FUNNEL_ORDER
                    if base.get(s, {}).get("grade") in ("ALERT", "WATCH")]

        if problems:
            first = problems[0]
            downstream = problems[1:]
            dive = tool_results.get(f"{first}_深挖", {})
            root = dive.get("root_cause") or base[first].get("root_cause", "")

            root_cause = f"根因：{first}环节 {base[first]['main_metric']}——{root}"
            if downstream:
                root_cause += f"；向下游传导至「{' → '.join(downstream)}」，形成连锁，最终导致 {symptom}。"
            else:
                root_cause += f"，最终导致 {symptom}。"

            warnings = []
            for s in FUNNEL_ORDER:
                if FUNNEL_ORDER.index(s) > FUNNEL_ORDER.index(first) and base.get(s, {}).get("grade") == "OK":
                    warnings.append(f"{s}环节虽健康，但受上游「{first}」隐患传导，需持续关注")
            if warnings:
                root_cause += "\n" + "\n".join(f"[!] {w}" for w in warnings)
        else:
            root_cause = f"各环节健康，{symptom}需排查外部因素（如流量大盘、竞争格局）。"

        actions = [f"[{'P0' if r.get('grade') == 'ALERT' else 'P1'}] {s}：{r.get('advice', '')}"
                   for s, r in sorted(base.items(), key=lambda kv: severity.get(kv[1].get("grade"), 9))
                   if r.get("grade") in ("ALERT", "WATCH")]

        result = root_cause + "\n" + "\n".join(actions)

        # RAG 注入：把知识库检索到的打法作为「外部知识参考」附在策略后
        if knowledge:
            result += "\n【知识库参考（RAG）】" + "；".join(knowledge)

        return result


class LLMBrain(Brain):
    """真实大模型大脑：推理与决策由 DeepSeek / OpenAI 兼容接口生成。"""

    def __init__(self, config: dict):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("使用真实大模型请先安装：pip install langchain-openai") from e
        api_key = (config.get("api_key")
                   or os.environ.get("DEEPSEEK_API_KEY")
                   or os.environ.get("OPENAI_API_KEY"))
        if not api_key:
            raise ValueError(
                "真实大模型需要 API Key：请在 config.py 的 LLM_CONFIG.api_key "
                "或环境变量 DEEPSEEK_API_KEY / OPENAI_API_KEY 中配置。"
            )
        self.llm = ChatOpenAI(
            model=config.get("model", "deepseek-chat"),
            api_key=api_key,
            base_url=config.get("base_url"),
            temperature=config.get("temperature", 0.0),
        )

    def _call(self, system: str, prompt: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        return resp.content

    def route_next(self, messages, tool_results, deep_dive):
        system = (
            "你是多智能体系统的统筹 Supervisor。根据已完成的专家诊断和对话历史，"
            "动态决定下一步：返回一个专家名（拉新/激活/留存/裂变/收入/拉新_深挖/…），"
            "或返回 FINISH 表示可以收尾。只返回决策词，不要解释。"
        )
        prompt = f"已完成：{list(tool_results.keys())}；深挖队列：{deep_dive}\n对话：{messages[-6:]}"
        raw = self._call(system, prompt).strip()
        return self._parse_route(raw, tool_results, deep_dive)

    @staticmethod
    def _parse_route(raw, tool_results, deep_dive):
        from .knowledge import ROUTE_ORDER
        if any(k in raw.upper() for k in ("FINISH", "结束", "收尾", "完成")):
            return "FINISH"
        for s in ROUTE_ORDER:
            if s in raw:
                return f"{s}_深挖" if "深挖" in raw else s
        done = set(tool_results.keys())
        for s in ROUTE_ORDER:
            if s not in done:
                return s
        return "FINISH"

    def diagnose(self, agent_name, persona, tool_result, context,
                 working=None, episodic=None, deep=False):
        system = persona
        if deep:
            system += " 本轮为「深挖轮」：请聚焦根因（为什么这个环节出问题），给出根因链与可落地的修复动作。"
        mem = f"\n你的工作记忆：{working}\n你的历史经验：{episodic}" if (working or episodic) else ""
        prompt = (f"工具计算结果：{tool_result}\n上下文：{context}{mem}\n"
                  f"请给出诊断（含评级、根因、建议）。")
        return self._call(system, prompt)

    def reflect(self, agent_name, tool_result):
        system = "你是一个严谨的分析 Agent。请对自己的诊断做自我反思：结论是否完整？是否遗漏了跨环节影响？置信度是否合理？"
        prompt = f"你的诊断结果：{tool_result}\n请给出 1-2 句自我反思。"
        return self._call(system, prompt)

    def synthesize(self, messages, tool_results, symptom="增长指标下滑",
                   knowledge=None):
        system = "你是统筹 Supervisor。基于各专家结论，产出根因链 + 综合增长策略。"
        kg = f"\n知识库参考：{knowledge}" if knowledge else ""
        prompt = f"专家结论：{tool_results}\n症状：{symptom}\n对话：{messages[-8:]}{kg}"
        return self._call(system, prompt)


def get_brain(config: dict) -> Brain:
    """工厂函数：根据配置返回对应的大脑实例。"""
    if config.get("LLM_MODE") in ("deepseek", "openai", "llm"):
        return LLMBrain(config.get("LLM_CONFIG", {}))
    return MockBrain()
