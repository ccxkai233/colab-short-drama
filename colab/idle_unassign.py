#!/usr/bin/env python3
"""
Colab 任务空闲监控与自动释放算力看门狗
职责：
  - 定期轮询本地 ComfyUI `/queue` 与 `/history` 接口。
  - 当检测到任务队列全部清空，且持续空闲超过 `idle_minutes` 时，
    自动调用 `google.colab.runtime.unassign()` 释放 GPU 计算单元，避免空转扣费。
  - 支持两种安全策略：
      1) require_activity=True（默认）：至少执行过 1 个任务后，队列清空并空闲满 N 分钟即释放；
         同时提供 max_no_job_minutes 兜底（若开机后一直没发任务超过上限也会释放）。
      2) require_activity=False：开机就绪后直接计时，空闲满 N 分钟即释放。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[idle_unassign {ts}] {msg}", flush=True)


def fetch_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def trigger_unassign(reason: str, log_file: Path | None = None, dry_run: bool = False) -> None:
    msg = f"🛑 触发自动释放算力：{reason}"
    log(msg)
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(f"[{datetime.now().isoformat()}] {msg}\n")
        except OSError:
            pass

    if dry_run:
        log("[dry-run] 跳过实际调用 google.colab.runtime.unassign()。")
        return

    try:
        from google.colab import runtime  # type: ignore

        log("正在调用 google.colab.runtime.unassign()...")
        runtime.unassign()
    except ImportError:
        log("⚠️ 当前不在 Google Colab 运行环境中，无法调用 runtime.unassign()。")


def run_watchdog(
    *,
    api_url: str = "http://127.0.0.1:8188",
    idle_minutes: float = 15.0,
    max_no_job_minutes: float = 60.0,
    poll_seconds: float = 15.0,
    require_activity: bool = True,
    log_file: Path | None = None,
    dry_run: bool = False,
) -> None:
    base = api_url.rstrip("/")
    log(
        f"启动空闲看门狗 | 目标: {base} | 任务后空闲释放: {idle_minutes}m | "
        f"无任务最长兜底: {max_no_job_minutes}m | require_activity={require_activity}"
    )

    # 等待 ComfyUI 就绪
    while fetch_json(f"{base}/queue") is None:
        time.sleep(5)
    log("ComfyUI API 已就绪，开始监控队列状态。")

    boot_time = time.monotonic()
    initial_history = fetch_json(f"{base}/history") or {}
    seen_history_count = len(initial_history)
    had_activity = not require_activity
    idle_since: float | None = time.monotonic() if had_activity else None

    while True:
        time.sleep(poll_seconds)
        queue_data = fetch_json(f"{base}/queue")
        if queue_data is None:
            log("⚠️ ComfyUI 暂时无响应，继续观察...")
            continue

        running = queue_data.get("queue_running", [])
        pending = queue_data.get("queue_pending", [])
        busy = bool(running or pending)

        if not had_activity:
            hist = fetch_json(f"{base}/history") or {}
            if busy or len(hist) > seen_history_count:
                had_activity = True
                log("检测到渲染任务活动，将在全部任务完成并持续空闲后自动释放算力。")

        now = time.monotonic()
        if busy:
            idle_since = None
            continue

        if had_activity:
            if idle_since is None:
                idle_since = now
                log(f"队列已清空，开始空闲倒计时（{idle_minutes} 分钟后自动释放算力）。")
            elapsed_min = (now - idle_since) / 60.0
            if elapsed_min >= idle_minutes:
                trigger_unassign(
                    f"所有队列任务已完成且持续空闲 {elapsed_min:.1f} 分钟（阈值 {idle_minutes} 分钟）",
                    log_file=log_file,
                    dry_run=dry_run,
                )
                return
        else:
            no_job_min = (now - boot_time) / 60.0
            if max_no_job_minutes > 0 and no_job_min >= max_no_job_minutes:
                trigger_unassign(
                    f"开机后持续 {no_job_min:.1f} 分钟未收到任何任务（兜底阈值 {max_no_job_minutes} 分钟）",
                    log_file=log_file,
                    dry_run=dry_run,
                )
                return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyUI 空闲自动释放 Colab 算力看门狗")
    parser.add_argument("--api-url", default="http://127.0.0.1:8188")
    parser.add_argument("--idle-minutes", type=float, default=15.0)
    parser.add_argument("--max-no-job-minutes", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--no-require-activity",
        action="store_true",
        help="无需等待首个任务，直接开始空闲计时",
    )
    parser.add_argument(
        "--log-file",
        default="/content/drive/MyDrive/ComfyUI_Work/logs/unassign.log",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_watchdog(
        api_url=args.api_url,
        idle_minutes=args.idle_minutes,
        max_no_job_minutes=args.max_no_job_minutes,
        poll_seconds=args.poll_seconds,
        require_activity=not args.no_require_activity,
        log_file=Path(args.log_file) if args.log_file else None,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
