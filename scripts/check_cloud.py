#!/usr/bin/env python3
"""
Short-Drama-Factory 云端 ComfyUI 连通性与远程体检脚本
职责：
  1. 探测 trycloudflare.com 地址是否在线、延迟与 GPU 显存/PyTorch 版本（/system_stats）。
  2. 检查队列状态（/queue），并支持远程中断卡死任务（--interrupt）或清空队列（--clear-queue）。
  3. 检查核心插件节点是否成功加载（LatentSync-Enhanced / Inpaint-CropAndStitch / Wan / FLUX）。
  4. 检查关键模型目录已就绪的权重文件（checkpoints / diffusion_models / text_encoders / vae / loras）。
  5. 检查最近任务历史中的执行报错堆栈（/history），便于快速定位节点错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


def load_dotenv_fallback(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_base_url(cli_url: str | None) -> str:
    load_dotenv_fallback()
    url = (cli_url or os.environ.get("COLAB_API_URL") or "").strip().rstrip("/")
    if not url or "replace-with-your-tunnel" in url:
        raise SystemExit(
            "❌ 未提供有效的云端 ComfyUI 地址。\n"
            "请通过 --url https://xxxx.trycloudflare.com 传入，或在项目根目录 .env 中设置 COLAB_API_URL。"
        )
    return url


def fetch_json(session: requests.Session, url: str, timeout: float = 15.0) -> Any:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def inspect_models(session: requests.Session, base_url: str) -> dict[str, list[str]]:
    folders = ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras")
    found: dict[str, list[str]] = {}
    for folder in folders:
        try:
            data = fetch_json(session, f"{base_url}/models/{folder}", timeout=10.0)
            if isinstance(data, list):
                found[folder] = data
        except Exception:
            found[folder] = []
    return found


def extract_recent_errors(history: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    items = list(history.items())[-limit:]
    for prompt_id, entry in reversed(items):
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            messages = status.get("messages", [])
            err_detail: dict[str, Any] = {"prompt_id": prompt_id, "messages": []}
            for msg_type, payload in messages:
                if msg_type == "execution_error" and isinstance(payload, dict):
                    err_detail["node_id"] = payload.get("node_id")
                    err_detail["node_type"] = payload.get("node_type")
                    err_detail["exception_type"] = payload.get("exception_type")
                    err_detail["exception_message"] = payload.get("exception_message")
                    tb = payload.get("traceback", [])
                    err_detail["traceback_tail"] = tb[-4:] if isinstance(tb, list) else tb
                else:
                    err_detail["messages"].append([msg_type, payload])
            errors.append(err_detail)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="云端 Colab ComfyUI 连通性体检与远程排障工具")
    parser.add_argument("--url", default=None, help="Colab cloudflared 地址（默认读 .env COLAB_API_URL）")
    parser.add_argument("--interrupt", action="store_true", help="远程中断当前正在执行的 ComfyUI 任务")
    parser.add_argument("--clear-queue", action="store_true", help="远程清空所有待执行队列任务")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出完整体检报告")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = resolve_base_url(args.url)
    session = requests.Session()
    session.headers.update({"User-Agent": "Short-Drama-Factory/0.1"})

    # 可选远程维护动作
    if args.interrupt:
        r = session.post(f"{base_url}/interrupt", timeout=15)
        r.raise_for_status()
        print("🛑 已发送中断当前任务指令 (/interrupt)。")
    if args.clear_queue:
        r = session.post(f"{base_url}/queue", json={"clear": True}, timeout=15)
        r.raise_for_status()
        print("🧹 已清空云端等待队列 (/queue clear)。")

    report: dict[str, Any] = {"base_url": base_url}

    # 1. 系统与 GPU 状态
    t0 = time.monotonic()
    try:
        stats = fetch_json(session, f"{base_url}/system_stats", timeout=15.0)
    except Exception as exc:
        print(f"❌ 无法连接到云端 ComfyUI ({base_url}): {exc}", file=sys.stderr)
        return 1
    latency_ms = round((time.monotonic() - t0) * 1000, 1)
    report["latency_ms"] = latency_ms
    report["system_stats"] = stats

    # 2. 队列状态
    queue = fetch_json(session, f"{base_url}/queue", timeout=15.0)
    report["queue"] = {
        "running": len(queue.get("queue_running", [])),
        "pending": len(queue.get("queue_pending", [])),
    }

    # 3. 核心节点探测
    obj_info = fetch_json(session, f"{base_url}/object_info", timeout=30.0)
    node_names = sorted(obj_info.keys()) if isinstance(obj_info, dict) else []
    latentsync_nodes = [n for n in node_names if "latentsync" in n.lower()]
    inpaint_nodes = [n for n in node_names if "inpaintcrop" in n.lower() or "inpaintstitch" in n.lower()]
    wan_nodes = [n for n in node_names if "wan" in n.lower()]
    report["nodes_check"] = {
        "total_nodes": len(node_names),
        "latentsync_loaded": bool(latentsync_nodes),
        "latentsync_nodes": latentsync_nodes[:10],
        "inpaint_crop_stitch_loaded": bool(inpaint_nodes),
        "inpaint_nodes": inpaint_nodes[:10],
        "wan_nodes_count": len(wan_nodes),
    }

    # 4. 已就绪模型扫描
    models = inspect_models(session, base_url)
    report["models"] = models

    # 5. 最近报错抓取
    try:
        history = fetch_json(session, f"{base_url}/history?max_items=10", timeout=15.0)
        report["recent_errors"] = extract_recent_errors(history if isinstance(history, dict) else {})
    except Exception:
        report["recent_errors"] = []

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    # 人类可读输出
    sys_info = stats.get("system", {})
    devices = stats.get("devices", [])
    print(f"✅ 云端 ComfyUI 在线 | 地址: {base_url} | 延迟: {latency_ms} ms")
    print(
        f"   ComfyUI: {sys_info.get('comfyui_version', 'unknown')} | "
        f"PyTorch: {sys_info.get('pytorch_version', 'unknown')} | "
        f"Python: {sys_info.get('python_version', '').split()[0]}"
    )
    for dev in devices:
        vram_total = dev.get("vram_total", 0) / (1024**3)
        vram_free = dev.get("vram_free", 0) / (1024**3)
        print(f"   GPU: {dev.get('name')} | 显存可用: {vram_free:.1f} / {vram_total:.1f} GB")

    print(
        f"\n📋 队列状态: 运行中 {report['queue']['running']} | 排队中 {report['queue']['pending']}"
    )

    nc = report["nodes_check"]
    print(f"\n🧩 核心插件节点检查（总节点数: {nc['total_nodes']}）:")
    print(
        f"   {'✅' if nc['latentsync_loaded'] else '❌'} LatentSync-Enhanced: "
        f"{', '.join(nc['latentsync_nodes']) if nc['latentsync_nodes'] else '未加载（请检查 Drive/logs/comfyui.log）'}"
    )
    print(
        f"   {'✅' if nc['inpaint_crop_stitch_loaded'] else '❌'} Inpaint-CropAndStitch: "
        f"{', '.join(nc['inpaint_nodes']) if nc['inpaint_nodes'] else '未加载'}"
    )
    print(f"   {'✅' if nc['wan_nodes_count'] > 0 else '⚠️'} Wan 原生节点数: {nc['wan_nodes_count']}")

    print("\n📦 云端已就绪模型:")
    for folder, files in models.items():
        preview = ", ".join(files[:6]) + (f" ... (+{len(files) - 6})" if len(files) > 6 else "")
        print(f"   - {folder:17s} ({len(files):2d}): {preview or '(空)'}")

    if report["recent_errors"]:
        print(f"\n❌ 检测到最近 {len(report['recent_errors'])} 条执行报错:")
        for err in report["recent_errors"]:
            print(
                f"   - prompt_id={err.get('prompt_id')} | 节点 {err.get('node_id')} ({err.get('node_type')}): "
                f"{err.get('exception_type')}: {err.get('exception_message')}"
            )
    else:
        print("\n✨ 最近历史记录中无执行报错。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
