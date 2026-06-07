"""
collect_results.py — 汇总各实验产生的 JSON（eval / benchmark / OOM 记录），
生成便于阅读的 Markdown 表格与 CSV，写入 results/summary.md 与 results/summary.csv。

约定（与 run_all.sh 的写出路径一致）：
    results/eval_*.json            —— eval_metrics.py 产出（含 model/n_nodes/mse/rne/...）
    results/benchmark_4000.json    —— benchmark_inference.py 产出（2.5）
    results/oom_*.json             —— run_all.sh 在训练 OOM 时写出的标记
    results/anchor_vs_temperature_*.json —— 2.6 定量数据（此处不汇总，仅图）

用法：
    python experiments/collect_results.py --results-dir results
"""

import os
import csv
import json
import glob
import argparse
import pathlib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    return p.parse_args()


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def fmt(x):
    if isinstance(x, (int, float)):
        return f"{x:.6g}"
    return str(x)


def main():
    args = parse_args()
    rd = pathlib.Path(args.results_dir)
    rows = []

    # eval_*.json
    for p in sorted(glob.glob(str(rd / "eval_*.json"))):
        d = load_json(p)
        if not d:
            continue
        rows.append({
            "experiment": pathlib.Path(p).stem.replace("eval_", ""),
            "model": d.get("model", ""),
            "n_nodes": d.get("n_nodes", ""),
            "split": d.get("split", ""),
            "mse": d.get("mse", ""),
            "rne": d.get("rne", ""),
            "mse_last": d.get("mse_last", ""),
            "rne_last": d.get("rne_last", ""),
            "status": "ok",
        })

    # oom_*.json（训练或评测阶段记录的 OOM）
    for p in sorted(glob.glob(str(rd / "oom_*.json"))):
        d = load_json(p) or {}
        rows.append({
            "experiment": pathlib.Path(p).stem.replace("oom_", ""),
            "model": d.get("model", ""),
            "n_nodes": d.get("n_nodes", ""),
            "split": d.get("split", ""),
            "mse": "OOM", "rne": "OOM", "mse_last": "OOM", "rne_last": "OOM",
            "status": "OOM",
        })

    # 写 CSV
    csv_path = rd / "summary.csv"
    cols = ["experiment", "model", "n_nodes", "split",
            "mse", "rne", "mse_last", "rne_last", "status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 写 Markdown
    md = ["# 实验结果汇总\n"]
    md.append("## 训练 / 评测指标（MSE / RNE）\n")
    md.append("| experiment | model | N | split | MSE | RNE | MSE(last) | RNE(last) | status |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append("| {experiment} | {model} | {n_nodes} | {split} | {mse} | {rne} | "
                  "{mse_last} | {rne_last} | {status} |".format(
                      experiment=r["experiment"], model=r["model"],
                      n_nodes=r["n_nodes"], split=r["split"],
                      mse=fmt(r["mse"]), rne=fmt(r["rne"]),
                      mse_last=fmt(r["mse_last"]), rne_last=fmt(r["rne_last"]),
                      status=r["status"]))

    # 2.5 推理基准（benchmark_inference.py 写出的是 {model: {...}} 字典）
    bench = load_json(rd / "benchmark_4000.json")
    if isinstance(bench, dict) and bench:
        md.append("\n## 2.5 推理基准（N=4000，未知轨迹）\n")
        md.append("| model | MSE | RNE | time/traj (s) | peak mem (MB) | status |")
        md.append("|---|---|---|---|---|---|")
        for model_name, e in bench.items():
            if not isinstance(e, dict):
                continue
            if "error" in e:
                md.append("| {m} | {st} | {st} | {st} | {st} | {st} |".format(
                    m=model_name, st=e.get("error", "error")))
            else:
                md.append("| {m} | {mse} | {rne} | {t} | {mem} | ok |".format(
                    m=model_name,
                    mse=fmt(e.get("mse", "")), rne=fmt(e.get("rne", "")),
                    t=fmt(e.get("time_per_traj_s", "")),
                    mem=fmt(e.get("peak_mem_MB", ""))))

    md_path = rd / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"[collect] 已写出 {md_path}")
    print(f"[collect] 已写出 {csv_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
