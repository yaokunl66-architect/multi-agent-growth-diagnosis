# -*- coding: utf-8 -*-
"""【数据】真实场景数据生成器：支持多场景，用于演示「跨会话记忆」。"""
import numpy as np


def _channels() -> list:
    """5 个渠道的投放数据（注意「信息流广告」ROI 仅 1.5，拉新质量差）。"""
    ch = [
        {"name": "信息流广告", "spend": 80000, "new_users": 4000, "gmv": 120000},
        {"name": "搜索广告", "spend": 60000, "new_users": 3000, "gmv": 180000},
        {"name": "社交裂变", "spend": 30000, "new_users": 5000, "gmv": 90000},
        {"name": "内容营销", "spend": 40000, "new_users": 2000, "gmv": 100000},
        {"name": "KOL 合作", "spend": 50000, "new_users": 2500, "gmv": 75000},
    ]
    for c in ch:
        c["roi"] = round(c["gmv"] / c["spend"], 2)
        c["cac"] = round(c["spend"] / c["new_users"], 2)
    return ch


def generate_data(scenario: str = "gmv_decline", seed: int = 42) -> dict:
    """生成 AARRR 数据 + 场景上下文。scenario 可选 gmv_decline / retention_crash。"""
    np.random.seed(seed)

    channels = _channels()
    activation = {"new_users": sum(c["new_users"] for c in channels),
                  "activated": 9800, "first_order": 7200}

    if scenario == "retention_crash":
        retention = {"day": [1, 7, 30], "rate": [0.25, 0.10, 0.04]}
        scenario_meta = {
            "platform": "某电商平台", "period": "2026 Q3 后期",
            "symptom": "次日留存率较基准腰斩",
            "task": "定位留存崩盘的根因链，并给出可落地的召回与留存策略",
        }
    else:  # 默认 gmv_decline
        retention = {"day": [1, 7, 30], "rate": [0.42, 0.22, 0.12]}
        scenario_meta = {
            "platform": "某电商平台", "period": "2026 Q3",
            "symptom": "GMV 环比下滑 12%",
            "task": "定位 GMV 下滑的根因链，并给出可落地的增长策略",
        }

    revenue = {"gmv": float(sum(c["gmv"] for c in channels)),
               "arppu": 260.0, "repurchase_rate": 0.28, "avg_lifecycle_days": 180.0}

    referral = {"share_rate": 0.18, "avg_invites": 3.5, "invite_conv": 0.12}

    return {
        "channels": channels,
        "activation": activation,
        "retention": retention,
        "revenue": revenue,
        "referral": referral,
        "scenario": scenario_meta,
    }
