# -*- coding: utf-8 -*-
"""【报告】把多智能体的最终结论，落成 JSON / TXT / PNG 三份交付物。"""
import json
import os

from .knowledge import GRADE_CN


def save_report(report: dict, out_dir: str = "output") -> dict:
    """保存诊断报告（JSON + TXT），并尝试生成 PNG 画像（需 matplotlib）。"""
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "multi_agent_report.json")
    txt_path = os.path.join(out_dir, "multi_agent_report.txt")
    png_path = os.path.join(out_dir, "multi_agent_report.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(_to_text(report))

    png_ok = False
    try:
        _to_png(report, png_path)
        png_ok = True
    except Exception:
        png_ok = False

    return {"json": json_path, "txt": txt_path,
            "png": png_path if png_ok else None}


def _to_text(report: dict) -> str:
    lines = []
    lines.append("=" * 52)
    lines.append(f"  {report.get('title', '多智能体诊断报告')}")
    lines.append("=" * 52)
    sc = report.get("scenario", {})
    lines.append(f"平台：{sc.get('platform', '')}  周期：{sc.get('period', '')}")
    lines.append(f"症状：{sc.get('symptom', '')}")
    lines.append(f"任务：{sc.get('task', '')}")
    lines.append("")
    lines.append("【参与 Agent】" + "、".join(report.get("agents", [])))
    lines.append("")
    lines.append("【健康度总览】")
    for k, v in report.get("health", {}).items():
        lines.append(f"  - {k}：{v.get('metric', '')} → {GRADE_CN.get(v.get('grade'), v.get('grade'))}")
    lines.append("")
    if report.get("deep_dive"):
        lines.append("【根因深挖】")
        for k, v in report["deep_dive"].items():
            lines.append(f"  - {k}：{v.get('root_cause', '')}")
        lines.append("")
    if report.get("reflections"):
        lines.append("【专家自我反思】")
        for r in report["reflections"]:
            lines.append(f"  - {r}")
        lines.append("")
    if report.get("knowledge"):
        method = report.get("retrieval_method", "keyword")
        lines.append(f"【知识库参考（RAG · {method}）】")
        for kg in report["knowledge"]:
            lines.append(f"  - {kg}")
        lines.append("")
    lines.append("【综合诊断 · 根因链 + 策略】")
    lines.append(report.get("synthesis", ""))
    lines.append("")
    lines.append("=" * 52)
    lines.append("【团队对话全程】")
    for m in report.get("conversation", []):
        lines.append("  " + m)
    return "\n".join(lines)


def _to_png(report: dict, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    health = report.get("health", {})
    names = list(health.keys())
    grade_order = {"ALERT": 3, "WATCH": 2, "OK": 1}
    scores = [grade_order.get(v.get("grade", "OK"), 1) for v in health.values()]
    colors = ["#d64545" if g == "ALERT" else "#e6a23c" if g == "WATCH" else "#67c23a"
              for g in [health[n]["grade"] for n in names]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, scores, color=colors)
    ax.set_ylabel("健康度（OK=1 WATCH=2 ALERT=3）")
    ax.set_title("多智能体 · 环节健康度画像")
    for i, (n, v) in enumerate(zip(names, scores)):
        ax.text(i, v + 0.05, health[n]["grade"], ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
