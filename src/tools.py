# -*- coding: utf-8 -*-
"""【第 3 步 · 工具】每个专家的「动手能力」（Function Calling）。

一句话理解：工具 = 专家的「手」，负责精确算数；大脑 = 专家的「脑」，负责做判断。
分工原因：CAC/ROI/K 因子这些指标都是公式，用代码算最准、最省 token；
「这个指标健康吗、该怎么办」才交给大脑（可换成真实大模型）去推理。

@tool 是 LangChain 的装饰器：把一个普通函数「包装」成能被 Agent 调用的工具
（Function Calling 协议），真实大模型模式下，LLM 可以决定何时调用它。
"""
from langchain_core.tools import tool

from .knowledge import THRESHOLDS, PLAYBOOKS


def _grade(stage: str, value: float) -> str:
    """按阈值给指标做「三级健康评级」：OK / WATCH / ALERT。"""
    th = THRESHOLDS[stage]
    if value >= th["ok_ge"]:
        return "OK"
    if value >= th["watch_ge"]:
        return "WATCH"
    return "ALERT"


def _advice(stage: str, grade: str) -> str:
    """根据环节 + 评级，从打法手册（PLAYBOOKS）取对应的策略建议。"""
    return PLAYBOOKS[stage]["recommend"].get(grade, "")


@tool
def analyze_acquisition(channels) -> dict:
    """【拉新工具】算整体 CAC、ROI，找出最佳/最差渠道。"""
    total_spend = float(sum(c["spend"] for c in channels))
    total_users = int(sum(c["new_users"] for c in channels))
    overall_cac = total_spend / total_users
    overall_roi = float(sum(c["gmv"] for c in channels)) / total_spend
    best = max(channels, key=lambda c: c["roi"])
    worst = min(channels, key=lambda c: c["roi"])
    grade = _grade("acquisition", overall_roi)

    root_cause = (
        f"低效渠道「{worst['name']}」ROI={worst['roi']}（低于健康线 2.0），"
        f"占比偏高导致拉新质量下降"
    ) if grade != "OK" else ""

    return {
        "stage": "拉新", "grade": grade,
        "main_metric": f"整体 ROI={overall_roi:.2f}",
        "extra": f"整体 CAC={overall_cac:.2f}；最佳渠道「{best['name']}」ROI={best['roi']}；"
                 f"最差渠道「{worst['name']}」ROI={worst['roi']}",
        "advice": _advice("acquisition", grade),
        "root_cause": root_cause,
        "overall_cac": round(overall_cac, 2), "overall_roi": round(overall_roi, 2),
        "best_channel": {"name": best["name"], "roi": best["roi"]},
        "worst_channel": {"name": worst["name"], "roi": worst["roi"]},
    }


@tool
def analyze_activation(activation) -> dict:
    """【激活工具】算激活率与首单转化率。"""
    act_rate = activation["activated"] / activation["new_users"]
    order_rate = activation["first_order"] / activation["new_users"]
    grade = _grade("activation", act_rate)
    root_cause = (
        f"激活率 {act_rate:.1%} 低于 60% 健康线，新手引导/首单路径存在流失点"
    ) if grade != "OK" else ""
    return {
        "stage": "激活", "grade": grade,
        "main_metric": f"激活率={act_rate:.3f}",
        "extra": f"首单转化率={order_rate:.3f}",
        "advice": _advice("activation", grade),
        "root_cause": root_cause,
        "activation_rate": round(act_rate, 3), "first_order_rate": round(order_rate, 3),
    }


@tool
def analyze_retention(retention) -> dict:
    """【留存工具】算 D1/D7/D30 留存与衰减比。"""
    r = retention["rate"]
    grade = _grade("retention", r[0])
    root_cause = (
        f"次日留存 {r[0]:.0%}、D7/D1 衰减比 {r[1]/r[0]:.2f}，产品价值感不足"
    ) if grade != "OK" else ""
    return {
        "stage": "留存", "grade": grade,
        "main_metric": f"次日留存={r[0]:.0%}",
        "extra": f"D7={r[1]:.0%} / D30={r[2]:.0%}；D7/D1 衰减比={r[1]/r[0]:.2f}",
        "advice": _advice("retention", grade),
        "root_cause": root_cause,
        "d1": r[0], "d7": r[1], "d30": r[2],
    }


@tool
def analyze_revenue(revenue, overall_cac) -> dict:
    """【收入工具】算 LTV 与 LTV/CAC（依赖拉新专家算出的整体 CAC）。"""
    arppu = revenue["arppu"]
    ltv = arppu * (revenue["avg_lifecycle_days"] / 30) * (1 + revenue["repurchase_rate"])
    ltv_cac = ltv / overall_cac if overall_cac > 0 else 0.0
    grade = _grade("revenue", ltv_cac)
    root_cause = (
        f"LTV/CAC={ltv_cac:.2f}，获客成本偏高、增长质量承压"
    ) if grade != "OK" else ""
    return {
        "stage": "收入", "grade": grade,
        "main_metric": f"LTV/CAC={ltv_cac:.2f}",
        "extra": f"LTV={ltv:.0f}；ARPPU={arppu}；复购率={revenue['repurchase_rate']}；"
                 f"（CAC 由拉新专家提供={overall_cac}）",
        "advice": _advice("revenue", grade),
        "root_cause": root_cause,
        "ltv": round(ltv, 2), "ltv_cac": round(ltv_cac, 2),
        "gmv": float(revenue["gmv"]), "arppu": arppu,
    }


@tool
def analyze_referral(referral) -> dict:
    """【裂变工具】算 K 因子（病毒系数，衡量自传播能力）。"""
    share = referral["share_rate"]
    invites = referral["avg_invites"]
    conv = referral["invite_conv"]
    k = share * invites * conv
    grade = _grade("referral", k)
    root_cause = (
        f"K 因子={k:.3f}（分享率{share}×人均邀请{invites}×邀请转化{conv}）偏低，"
        f"自然传播几乎失效、增长依赖付费投放"
    ) if grade != "OK" else ""
    return {
        "stage": "裂变", "grade": grade,
        "main_metric": f"K 因子={k:.3f}",
        "extra": f"分享率={share}；人均邀请={invites}；邀请转化={conv}",
        "advice": _advice("referral", grade),
        "root_cause": root_cause,
        "k_factor": round(k, 3),
        "share_rate": share, "avg_invites": invites, "invite_conv": conv,
    }


# 工具注册表：把 5 个工具按「环节中文名」登记，供专家节点按名查找
TOOLS = {
    "拉新": analyze_acquisition,
    "激活": analyze_activation,
    "留存": analyze_retention,
    "收入": analyze_revenue,
    "裂变": analyze_referral,
}
