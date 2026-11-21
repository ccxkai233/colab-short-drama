#!/usr/bin/env python3
"""
Short-Drama-Factory 云端 ComfyUI 连通性与远程体检脚本（支持单实例 & 多实例并发池）
职责：
  1. 探测单个或多个 trycloudflare.com 地址是否在线、延迟与 GPU 显存/PyTorch 版本（/system_stats）。
  2. 检查队列状态（/queue），并支持远程中断卡死任务（--interrupt）或清空队列（--clear-queue）。
  3. 检查核心原生与插件节点是否加载（MiniMax H3 / LatentSync-Enhanced / Inpaint-CropAndStitch / Wan）。
  4. 检查关键模型目录已就绪的权重文件（checkpoints / diffusion_models / text_encoders / vae / loras / upscale_models）。
  5. 检查最近任务历史中的执行报错堆栈（/history），便于快速定位节点错误。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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


def resolve_base_urls(cli_urls: str | None, cli_url: str | None) -> list[str]:
    load_dotenv_fallback()
    raw = cli_urls or cli_url or os.environ.get("COLAB_API_URLS") or os.environ.get("COLAB_API_URL") or ""
    urls: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;\s]+", raw.strip()):
        u = part.strip().rstrip("/")
        if not u or "replace-with-your-tunnel" in u:
            continue
        if u not in seen:
            seen.add(u)
            urls.append(u)
    if not urls:
        raise SystemExit(
            "❌ 未提供有效的云端 ComfyUI 地址。\n"
            "请通过 --url 或 --urls 传入，或在项目根目录 .env 中设置 COLAB_API_URL / COLAB_API_URLS。"
        )
    return urls


def fetch_json(session: requests.Session, url: str, timeout: float = 15.0) -> Any:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def inspect_models(session: requests.Session, base_url: str) -> dict[str, list[str]]:
    folders = ("checkpoints", "diffusion_models", "text_encoders", "vae", "loras", "upscale_models")
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


def inspect_single_node(base_url: str, *, do_interrupt: bool = False, do_clear_queue: bool = False) -> dict[str, Any]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Short-Drama-Factory/0.2"})
    report: dict[str, Any] = {"base_url": base_url, "online": False, "actions": []}

    if do_interrupt:
        try:
            r = session.post(f"{base_url}/interrupt", timeout=15)
            r.raise_for_status()
            report["actions"].append("interrupted")
        except Exception as exc:
            report["actions"].append(f"interrupt_failed: {exc}")
    if do_clear_queue:
        try:
            r = session.post(f"{base_url}/queue", json={"clear": True}, timeout=15)
            r.raise_for_status()
            report["actions"].append("queue_cleared")
        except Exception as exc:
            report["actions"].append(f"clear_queue_failed: {exc}")

    t0 = time.monotonic()
    try:
        stats = fetch_json(session, f"{base_url}/system_stats", timeout=15.0)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    report["online"] = True
    report["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
    report["system_stats"] = stats

    queue_data = fetch_json(session, f"{base_url}/queue", timeout=15.0)
    report["queue"] = {
        "running": len(queue_data.get("queue_running", [])),
        "pending": len(queue_data.get("queue_pending", [])),
    }

    obj_info = fetch_json(session, f"{base_url}/object_info", timeout=30.0)
    node_names = sorted(obj_info.keys()) if isinstance(obj_info, dict) else []
    minimax_h3_nodes = [n for n in node_names if "minimaxh3" in n.lower()]
    has_audio_vae_decode = "VAEDecodeAudio" in obj_info if isinstance(obj_info, dict) else False
    latentsync_nodes = [n for n in node_names if "latentsync" in n.lower()]
    inpaint_nodes = [n for n in node_names if "inpaintcrop" in n.lower() or "inpaintstitch" in n.lower()]
    wan_nodes = [n for n in node_names if "wan" in n.lower()]
    report["nodes_check"] = {
        "total_nodes": len(node_names),
        "minimax_h3_loaded": bool(minimax_h3_nodes) and has_audio_vae_decode,
        "minimax_h3_nodes": minimax_h3_nodes[:12],
        "vae_decode_audio_loaded": has_audio_vae_decode,
        "latentsync_loaded": bool(latentsync_nodes),
        "latentsync_nodes": latentsync_nodes[:10],
        "inpaint_crop_stitch_loaded": bool(inpaint_nodes),
        "inpaint_nodes": inpaint_nodes[:10],
        "wan_nodes_count": len(wan_nodes),
    }

    report["models"] = inspect_models(session, base_url)

    try:
        history = fetch_json(session, f"{base_url}/history?max_items=10", timeout=15.0)
        report["recent_errors"] = extract_recent_errors(history if isinstance(history, dict) else {})
    except Exception:
        report["recent_errors"] = []

    return report


def print_node_report(report: dict[str, Any], label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    base_url = report["base_url"]
    for act in report.get("actions", []):
        if act == "interrupted":
            print(f"{prefix}🛑 已发送中断当前任务指令 (/interrupt)。")
        elif act == "queue_cleared":
            print(f"{prefix}🧹 已清空云端等待队列 (/queue clear)。")
        else:
            print(f"{prefix}⚠️ 维护动作异常: {act}")

    if not report.get("online"):
        print(f"{prefix}❌ 无法连接到云端 ComfyUI ({base_url}): {report.get('error')}", file=sys.stderr)
        return

    stats = report["system_stats"]
    sys_info = stats.get("system", {})
    devices = stats.get("devices", [])
    print(f"{prefix}✅ 云端 ComfyUI 在线 | 地址: {base_url} | 延迟: {report['latency_ms']} ms")
    print(
        f"   ComfyUI: {sys_info.get('comfyui_version', 'unknown')} | "
        f"PyTorch: {sys_info.get('pytorch_version', 'unknown')} | "
        f"Python: {sys_info.get('python_version', '').split()[0]}"
    )
    for dev in devices:
        vram_total = dev.get("vram_total", 0) / (1024**3)
        vram_free = dev.get("vram_free", 0) / (1024**3)
        print(f"   GPU: {dev.get('name')} | 显存可用: {vram_free:.1f} / {vram_total:.1f} GB")

    print(f"\n📋 队列状态: 运行中 {report['queue']['running']} | 排队中 {report['queue']['pending']}")

    nc = report["nodes_check"]
    print(f"\n🧩 核心插件与原生节点检查（总节点数: {nc['total_nodes']}）:")
    print(
        f"   {'✅' if nc['minimax_h3_loaded'] else '❌'} MiniMax H3 原生音画节点: "
        f"{', '.join(nc['minimax_h3_nodes']) if nc['minimax_h3_nodes'] else '未检测到（请升级 ComfyUI >= 0.45）'}"
    )
    print(
        f"   {'✅' if nc['latentsync_loaded'] else '⚠️'} LatentSync-Enhanced (备用): "
        f"{', '.join(nc['latentsync_nodes']) if nc['latentsync_nodes'] else '未加载（如仅用 MiniMax H3 可忽略）'}"
    )
    print(
        f"   {'✅' if nc['inpaint_crop_stitch_loaded'] else '❌'} Inpaint-CropAndStitch: "
        f"{', '.join(nc['inpaint_nodes']) if nc['inpaint_nodes'] else '未加载'}"
    )
    print(f"   {'✅' if nc['wan_nodes_count'] > 0 else '⚠️'} Wan 原生节点数: {nc['wan_nodes_count']}")

    print("\n📦 云端已就绪模型:")
    for folder, files in report["models"].items():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="云端 Colab ComfyUI 连通性体检与远程排障工具（支持多实例）")
    parser.add_argument("--url", default=None, help="单个 Colab cloudflared 地址")
    parser.add_argument("--urls", default=None, help="多个云端地址（逗号分隔，并发体检）")
    parser.add_argument("--interrupt", action="store_true", help="远程中断当前正在执行的 ComfyUI 任务")
    parser.add_argument("--clear-queue", action="store_true", help="远程清空所有待执行队列任务")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出完整体检报告")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    urls = resolve_base_urls(args.urls, args.url)

    with ThreadPoolExecutor(max_workers=max(1, len(urls))) as pool:
        reports = list(
            pool.map(
                lambda u: inspect_single_node(u, do_interrupt=args.interrupt, do_clear_queue=args.clear_queue),
                urls,
            )
        )

    if args.json:
        out: Any = reports[0] if len(reports) == 1 else {"nodes": reports}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if any(r["online"] for r in reports) else 1

    for idx, rep in enumerate(reports, start=1):
        if len(reports) > 1:
            print(f"\n{'=' * 68}")
            print_node_report(rep, label=f"W{idx}")
        else:
            print_node_report(rep)

    if len(reports) > 1:
        online = [r for r in reports if r["online"]]
        total_vram_free = sum(
            dev.get("vram_free", 0) / (1024**3)
            for r in online
            for dev in r.get("system_stats", {}).get("devices", [])
        )
        total_vram_all = sum(
            dev.get("vram_total", 0) / (1024**3)
            for r in online
            for dev in r.get("system_stats", {}).get("devices", [])
        )
        total_running = sum(r["queue"]["running"] for r in online)
        total_pending = sum(r["queue"]["pending"] for r in online)
        print(f"\n{'=' * 68}")
        print(
            f"🌐 多实例集群汇总: 在线 {len(online)}/{len(reports)} 台 | "
            f"总可用显存 {total_vram_free:.1f} / {total_vram_all:.1f} GB | "
            f"运行中 {total_running} | 排队中 {total_pending}"
        )

    return 0 if any(r["online"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
