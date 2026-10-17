# -*- coding: utf-8 -*-
"""【记忆系统】每个 Agent 独立的记忆：工作记忆（短期）+ 情节记忆（长期）。

一句话理解：真·多智能体的每个 Agent，不是"一次性算完就走"，而是有自己的"记性"——
- 工作记忆（working，短期）：本次任务中的笔记、发现、思考草稿，任务结束即清空；
- 情节记忆（episodic，长期）：跨会话积累的历史诊断经验，下次运行能"回忆"起来。

为什么这很重要？—— 单次诊断的 Agent 是"失忆"的：每次从零开始，不记得上次
"裂变告急时推荐过拼团玩法"。有记忆的 Agent 第二次遇到类似问题，能直接复用经验，
这才是"会成长的智能体"，也是本课与「多 worker 并行」的又一本质区别。

本版升级：情节记忆检索从「关键词重合度」升级为 **ChromaDB 向量检索**——
commit() 时把工作记忆写入 ChromaDB 集合，recall_episodic() 用向量相似度召回
（保留关键词兜底），实现真正的「语义记忆」。传入 base_dir 时用持久化客户端，
跨「次运行」也能记住经验。
"""
import os
import uuid
from dataclasses import dataclass, field


@dataclass
class Memory:
    """一个 Agent 的记忆单元（每个 sub-agent 独立的私有状态）。

    情节记忆双写：既存内存列表（快照/兜底用），又写 ChromaDB 集合（向量检索用）。
    base_dir 不为 None 时用 PersistentClient 落盘，跨运行持久化。
    """

    agent_name: str                              # 属于哪个专家（如"拉新"）
    base_dir: str = None                         # 持久化目录（None = 纯内存，不落盘）
    working: list = field(default_factory=list)  # 工作记忆（短期，本次任务内的笔记）
    episodic: list = field(default_factory=list) # 情节记忆（长期，跨会话积累的经验，内存副本）
    _collection: object = field(default=None, init=False, repr=False)  # ChromaDB 集合

    def __post_init__(self):
        self._init_chroma()
        self._load_from_disk()

    def _init_chroma(self):
        """为这个 Agent 建一个专属 ChromaDB 集合（失败则回退纯内存关键词检索）。"""
        try:
            import chromadb
            if self.base_dir:
                os.makedirs(self.base_dir, exist_ok=True)
                client = chromadb.PersistentClient(path=self.base_dir)
            else:
                client = chromadb.Client()
            self._collection = client.get_or_create_collection(f"memory_{self.agent_name}")
        except Exception:
            self._collection = None

    def _load_from_disk(self):
        """启动时把已持久化的历史情节记忆读回内存副本（快照/兜底用）。"""
        if self._collection is not None:
            try:
                existing = self._collection.get()
                self.episodic = list(existing.get("documents", []))
            except Exception:
                pass

    # ============ 写入 ============
    def note(self, text: str) -> None:
        """往工作记忆里记一条（本次任务中的思考 / 发现 / 结论）。"""
        self.working.append(text)

    # ============ 回忆 ============
    def recall_working(self, n: int = 5) -> list:
        """回忆最近 n 条工作记忆（= 本次任务到目前为止的上下文）。"""
        return self.working[-n:]

    def recall_episodic(self, query: str, top_k: int = 2) -> list:
        """从情节记忆里检索相关历史经验（跨会话回忆）。

        主通道：ChromaDB 向量检索（语义相似度召回）；
        兜底：关键词重合度打分（离线无网络时仍可用）。
        """
        if self._collection is not None:
            try:
                res = self._collection.query(query_texts=[query], n_results=top_k)
                docs = res.get("documents", [[]])[0]
                if docs:
                    return list(docs)
            except Exception:
                pass
        return self._keyword_episodic(query, top_k)

    def _keyword_episodic(self, query: str, top_k: int = 2) -> list:
        q = set(query)
        scored = []
        for e in self.episodic:
            score = len(q & set(e))
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    # ============ 沉淀 ============
    def commit(self) -> list:
        """任务结束：把工作记忆「沉淀」到情节记忆（形成长期经验），并清空工作记忆。

        这是「短期 → 长期」的转化点：本次的发现变成下次可复用的经验。
        沉淀时同步写入 ChromaDB（语义记忆）+ 内存列表（快照/兜底）。
        """
        committed = list(self.working)
        for text in committed:
            self.episodic.append(text)
            if self._collection is not None:
                try:
                    self._collection.upsert(ids=[uuid.uuid4().hex], documents=[text])
                except Exception:
                    pass
        self.working = []
        return committed

    # ============ 展示 ============
    def snapshot(self) -> dict:
        """导出记忆快照（供报告展示记忆系统的工作状态）。"""
        return {
            "agent": self.agent_name,
            "working": list(self.working),      # 本次任务的工作记忆
            "episodic": list(self.episodic),    # 长期情节记忆
        }


def build_memories(stages: list, base_dir: str = None) -> dict:
    """为每个专家 Agent 创建一个独立的 Memory 实例。

    这些 Memory 对象创建在「图之外」，跨 app.invoke() 复用，
    从而实现「跨会话记忆」——第二次运行能回忆起第一次的经验。
    传入 base_dir 时用持久化客户端，跨「次运行」也生效。
    """
    return {s: Memory(agent_name=s, base_dir=base_dir) for s in stages}
