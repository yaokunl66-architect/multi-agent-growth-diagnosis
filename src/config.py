# -*- coding: utf-8 -*-
"""【配置】全局配置中心：大脑模式、记忆持久化、RAG 知识库开关、路径。"""
import os

# ─────────────────────────────────────────────────────────────
# 大脑模式
#   "mock"      ：确定性 MockBrain（零 API Key、零网络、可复现，教学/离线首选）
#   "deepseek"  ：接 DeepSeek 官方接口（需 API Key）
#   "openai"    ：接 OpenAI 兼容接口（需 API Key + base_url）
# ─────────────────────────────────────────────────────────────
LLM_MODE = os.environ.get("MUTI_AGENT_LLM_MODE", "mock")

LLM_CONFIG = {
    "model": os.environ.get("MUTI_AGENT_MODEL", "deepseek-chat"),
    "api_key": os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", ""),
    "base_url": os.environ.get("OPENAI_BASE_URL", ""),
    "temperature": 0.0,
}

# ─────────────────────────────────────────────────────────────
# 记忆
# ─────────────────────────────────────────────────────────────
MEMORY_DIR = os.environ.get("MUTI_AGENT_MEMORY_DIR", "memory")  # 工作记忆 + 情节记忆(ChromaDB) 持久化目录

# ─────────────────────────────────────────────────────────────
# RAG 知识库（ChromaDB）
# ─────────────────────────────────────────────────────────────
RAG_DIR = os.environ.get("MUTI_AGENT_RAG_DIR", "rag_db")        # ChromaDB 持久化目录
RAG_TOP_K = int(os.environ.get("MUTI_AGENT_RAG_TOP_K", "3"))    # 每次检索返回条数

# ─────────────────────────────────────────────────────────────
# 运行
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
SCENARIO = os.environ.get("MUTI_AGENT_SCENARIO", "gmv_decline")  # gmv_decline / retention_crash
TASK = "诊断电商增长问题并给出可落地的增长策略"

# ─────────────────────────────────────────────────────────────
# 确保目录存在（内存库启动即可用，避免首次运行报错）
# ─────────────────────────────────────────────────────────────
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
