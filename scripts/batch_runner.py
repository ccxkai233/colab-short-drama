#!/usr/bin/env python3
"""
Short-Drama-Factory 本地 API 批量调度引擎
职责：
  1. 读取本地分镜表（CSV 或 JSON）与节点映射配置（workflow_bindings.yaml）。
  2. 加载对应任务类型的 ComfyUI API 格式工作流 JSON，自动替换 Prompt / Seed / 尺寸 / 输入文件等参数。
  3. 自动将本地素材通过 ComfyUI `/upload/image` 上传至云端 input/（同文件按指纹缓存，避免重复上传），
     亦支持 `--no-upload` 直接引用已同步在 Google Drive 的文件名。
  4. 提交任务到 Colab 暴露的 `trycloudflare.com` 地址排队渲染，优先使用 WebSocket 实时监听进度，
     断线或未安装 websocket-client 时自动无感降级为 HTTP `/history/{prompt_id}` 轮询。
  5. 渲染完成后自动通过 `/view` 拉回成片到本地 `workspace/output/<shot_id>/` 归档，
     并持久化断点状态 `workspace/logs/batch_state.json`，支持断点续跑与失败重试。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

try:
    import websocket  # type: ignore

    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False


def load_dotenv_fallback(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def cast_value(raw: Any, value_type: str | None) -> Any:
    if value_type is None:
        return raw
    t = value_type.lower()
    if t == "int":
        return int(float(raw))
    if t == "float":
        return float(raw)
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
    return str(raw)


def load_storyboard(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"❌ 分镜表不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "shots" in data:
            data = data["shots"]
        if not isinstance(data, list):
            raise SystemExit("❌ JSON 分镜表必须为镜头对象列表（或包含 shots 列表字段）。")
        return [dict(item) for item in data]
    if suffix == ".csv":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cleaned = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
                if any(cleaned.values()):
                    rows.append(cleaned)
        return rows
    raise SystemExit(f"❌ 不支持的分镜表格式: {suffix}（请使用 .csv 或 .json）")


def load_bindings(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"❌ 工作流绑定配置不存在: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "tasks" not in cfg:
        raise SystemExit(f"❌ 绑定配置格式错误，缺少顶层 tasks 字段: {path}")
    return cfg["tasks"]


def resolve_local_media(value: str, media_dir: Path) -> Path:
    direct = Path(value)
    if direct.exists():
        return direct.resolve()
    candidates = [
        media_dir / value,
        media_dir / "audio" / value,
        media_dir / "video" / value,
        media_dir / "images" / value,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        f"找不到本地素材文件 '{value}'（已检查原路径与 {media_dir} 下的 audio/video/images 子目录）"
    )


class ComfyBatchClient:
    def __init__(self, base_url: str, *, poll_interval: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.client_id = f"sdf-{uuid.uuid4().hex[:10]}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Short-Drama-Factory/0.1"})
        self._upload_cache: dict[tuple[str, float, int], str] = {}

    def upload_media(self, local_path: Path) -> str:
        st = local_path.stat()
        cache_key = (str(local_path), st.st_mtime, st.st_size)
        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        with local_path.open("rb") as fh:
            files = {"image": (local_path.name, fh, "application/octet-stream")}
            data = {"type": "input", "overwrite": "true"}
            resp = self.session.post(
                f"{self.base_url}/upload/image",
                files=files,
                data=data,
                timeout=300,
            )
            resp.raise_for_status()
            payload = resp.json()

        remote_name = payload.get("name", local_path.name)
        subfolder = payload.get("subfolder", "")
        final_ref = f"{subfolder}/{remote_name}" if subfolder else remote_name
        self._upload_cache[cache_key] = final_ref
        return final_ref

    def submit_prompt(self, workflow_api: dict[str, Any]) -> str:
        resp = self.session.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow_api, "client_id": self.client_id},
            timeout=60,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"提交 /prompt 失败 (HTTP {resp.status_code}): {resp.text[:500]}")
        data = resp.json()
        if data.get("node_errors"):
            raise RuntimeError(f"ComfyUI 节点参数校验报错: {json.dumps(data['node_errors'], ensure_ascii=False)}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"响应中缺少 prompt_id: {data}")
        return str(prompt_id)

    def _ws_url(self) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{ws_scheme}://{parsed.netloc}{parsed.path}/ws?clientId={self.client_id}"

    def wait_for_completion(self, prompt_id: str, timeout_sec: float = 1200.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec

        # 1) 优先用 WebSocket 实时监听进度，若断线或未装库则平滑降级到 HTTP 轮询
        if HAS_WEBSOCKET:
            try:
                ws = websocket.create_connection(self._ws_url(), timeout=15)
                try:
                    while time.monotonic() < deadline:
                        ws.settimeout(min(15.0, max(1.0, deadline - time.monotonic())))
                        try:
                            raw = ws.recv()
                        except websocket.WebSocketTimeoutException:
                            # 偶尔核对一次 /history 防止漏掉已完成事件
                            hist = self._fetch_prompt_history(prompt_id)
                            if hist is not None:
                                return hist
                            continue
                        if not isinstance(raw, str):
                            continue
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        mdata = msg.get("data", {})
                        if mdata.get("prompt_id") != prompt_id:
                            continue
                        if mtype == "progress":
                            val, mx = mdata.get("value", 0), mdata.get("max", 0)
                            if mx:
                                print(f"    ⏳ 渲染进度: {val}/{mx}", end="\r", flush=True)
                        elif mtype == "execution_error":
                            print("", flush=True)
                            raise RuntimeError(
                                f"节点 {mdata.get('node_id')} ({mdata.get('node_type')}) 执行异常: "
                                f"{mdata.get('exception_type')}: {mdata.get('exception_message')}"
                            )
                        elif mtype == "executing" and mdata.get("node") is None:
                            print("", flush=True)
                            hist = self._fetch_prompt_history(prompt_id)
                            if hist is not None:
                                return hist
                            break
                finally:
                    ws.close()
            except RuntimeError:
                raise
            except Exception as exc:
                print(f"    ⚠️ WebSocket 监听中断，自动切换为 HTTP 轮询 ({type(exc).__name__})", flush=True)

        # 2) HTTP /history 轮询兜底
        while time.monotonic() < deadline:
            hist = self._fetch_prompt_history(prompt_id)
            if hist is not None:
                return hist
            time.sleep(self.poll_interval)

        raise TimeoutError(f"等待任务 {prompt_id} 完成超时（>{timeout_sec:.0f}s）")

    def _fetch_prompt_history(self, prompt_id: str) -> dict[str, Any] | None:
        resp = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if prompt_id not in data:
            return None
        entry = data[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            for msg_type, payload in status.get("messages", []):
                if msg_type == "execution_error" and isinstance(payload, dict):
                    raise RuntimeError(
                        f"节点 {payload.get('node_id')} ({payload.get('node_type')}) 执行失败: "
                        f"{payload.get('exception_type')}: {payload.get('exception_message')}"
                    )
            raise RuntimeError(f"任务 {prompt_id} 执行失败: {status}")
        if status.get("completed") or entry.get("outputs"):
            return entry
        return None

    def download_outputs(
        self,
        history_entry: dict[str, Any],
        shot_output_dir: Path,
        shot_id: str,
        task_type: str,
    ) -> list[str]:
        shot_output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        outputs = history_entry.get("outputs", {})

        for _node_id, node_out in outputs.items():
            if not isinstance(node_out, dict):
                continue
            for media_key in ("videos", "gifs", "images", "audio"):
                items = node_out.get(media_key, [])
                if not isinstance(items, list):
                    continue
                for idx, item in enumerate(items):
                    if not isinstance(item, dict) or "filename" not in item:
                        continue
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    ftype = item.get("type", "output")
                    params = {"filename": filename, "subfolder": subfolder, "type": ftype}
                    resp = self.session.get(
                        f"{self.base_url}/view",
                        params=params,
                        stream=True,
                        timeout=300,
                    )
                    resp.raise_for_status()

                    safe_orig = Path(filename).name
                    local_name = (
                        safe_orig
                        if safe_orig.startswith(f"{shot_id}_")
                        else f"{shot_id}_{task_type}_{idx:02d}_{safe_orig}"
                    )
                    dst = shot_output_dir / local_name
                    with dst.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 256):
                            if chunk:
                                fh.write(chunk)
                    saved_paths.append(str(dst))
        return saved_paths


def build_prompt_payload(
    shot: dict[str, Any],
    task_cfg: dict[str, Any],
    *,
    client: ComfyBatchClient | None,
    media_dir: Path,
    no_upload: bool,
) -> tuple[dict[str, Any], int]:
    wf_path = Path(task_cfg["workflow"])
    if not wf_path.exists():
        raise FileNotFoundError(f"工作流 API JSON 文件不存在: {wf_path}")
    wf = copy.deepcopy(json.loads(wf_path.read_text(encoding="utf-8")))

    # 解析或生成可复现随机 Seed
    raw_seed = shot.get("seed", "")
    if raw_seed in ("", None, -1, "-1"):
        resolved_seed = random.randint(1, 2**53 - 1)
    else:
        resolved_seed = int(float(raw_seed))

    shot_id = str(shot["shot_id"])
    task_type = str(shot["task_type"])

    # 自动设置输出文件名前缀，方便 Drive output/ 与本地归档双向对应
    prefix_cfg = task_cfg.get("output_prefix_node")
    if prefix_cfg:
        pid = str(prefix_cfg["node_id"])
        pkey = str(prefix_cfg["input"])
        if pid in wf and "inputs" in wf[pid]:
            wf[pid]["inputs"][pkey] = f"{shot_id}_{task_type}"

    # 按绑定配置替换各节点字段
    bindings: dict[str, Any] = task_cfg.get("bindings", {})
    for field_name, bcfg in bindings.items():
        if field_name == "seed":
            val: Any = resolved_seed
        else:
            val = shot.get(field_name)
            if val is None or val == "":
                continue

        if bcfg.get("upload"):
            if no_upload or client is None:
                val = Path(str(val)).name
            else:
                local_file = resolve_local_media(str(val), media_dir)
                print(f"    ⬆️  上传素材: {local_file.name}", flush=True)
                val = client.upload_media(local_file)

        val = cast_value(val, bcfg.get("type"))
        node_ids = bcfg["node_id"]
        if not isinstance(node_ids, list):
            node_ids = [node_ids]
        input_key = str(bcfg["input"])

        for nid in node_ids:
            nid_str = str(nid)
            if nid_str not in wf:
                raise KeyError(
                    f"工作流 {wf_path} 中不存在节点 ID '{nid_str}'（字段 '{field_name}'），请检查 workflow_bindings.yaml。"
                )
            wf[nid_str].setdefault("inputs", {})[input_key] = val

    return wf, resolved_seed


def load_state(state_path: Path) -> dict[str, Any]:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"shots": {}}


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    load_dotenv_fallback()
    parser = argparse.ArgumentParser(description="Short-Drama-Factory 云端 ComfyUI API 批量调度脚本")
    parser.add_argument("--storyboard", required=True, help="分镜表路径（.csv 或 .json）")
    parser.add_argument(
        "--bindings",
        default=os.environ.get("WORKFLOW_BINDINGS_FILE", "configs/workflow_bindings.example.yaml"),
        help="工作流节点映射配置文件",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("COLAB_API_URL"),
        help="Colab cloudflared 地址（默认读 .env COLAB_API_URL）",
    )
    parser.add_argument("--media-dir", default="workspace/clean_input", help="本地清洗后素材目录")
    parser.add_argument("--output-dir", default="workspace/output", help="成片自动拉回归档目录")
    parser.add_argument("--state-file", default="workspace/logs/batch_state.json", help="断点续跑状态文件")
    parser.add_argument("--only-shots", default="", help="仅执行指定 shot_id（逗号分隔，如 S001,S002）")
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("JOB_TIMEOUT_SECONDS", "1200")),
        help="单镜头超时秒数",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("JOB_MAX_RETRIES", "2")),
        help="单镜头失败重试次数",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("POLL_INTERVAL_SECONDS", "5")),
        help="HTTP 轮询间隔秒数",
    )
    parser.add_argument("--no-upload", action="store_true", help="跳过素材上传（直接引用云端 Drive input/ 中已有文件名）")
    parser.add_argument("--force", action="store_true", help="忽略断点记录，强制重跑所有镜头")
    parser.add_argument("--retry-failed", action="store_true", help="仅重跑上次状态为 failed 的镜头")
    parser.add_argument("--dry-run", action="store_true", help="离线演练：生成替换后的 API JSON 到 workspace/logs/dry_run/，不发网络请求")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shots = load_storyboard(Path(args.storyboard))
    tasks_cfg = load_bindings(Path(args.bindings))
    media_dir = Path(args.media_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    state_path = Path(args.state_file).resolve()
    state = load_state(state_path)

    only_set = {s.strip() for s in args.only_shots.split(",") if s.strip()}

    client: ComfyBatchClient | None = None
    if not args.dry_run:
        url = (args.url or "").strip().rstrip("/")
        if not url or "replace-with-your-tunnel" in url:
            raise SystemExit(
                "❌ 未配置有效的 COLAB_API_URL。请通过 --url 传入或在 .env 中设置（离线演练可加 --dry-run）。"
            )
        client = ComfyBatchClient(url, poll_interval=args.poll_interval)
        print(f"🚀 已绑定云端渲染节点: {url} | 监听模式: {'WebSocket+HTTP兜底' if HAS_WEBSOCKET else 'HTTP轮询'}")
    else:
        print("🔍 [DRY-RUN 模式] 仅验证参数替换与工作流映射，不向云端发送请求。")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for idx, shot in enumerate(shots, start=1):
        shot_id = str(shot.get("shot_id") or f"SHOT_{idx:03d}")
        shot["shot_id"] = shot_id
        task_type = str(shot.get("task_type") or "").strip()

        if only_set and shot_id not in only_set:
            continue

        prev = state["shots"].get(shot_id, {})
        prev_status = prev.get("status")
        if args.retry_failed and prev_status != "failed":
            skip_count += 1
            continue
        if not args.force and prev_status == "done":
            print(f"⏭️  [{idx}/{len(shots)}] {shot_id} ({task_type}) 已完成，跳过（加 --force 可重跑）。")
            skip_count += 1
            continue

        if task_type not in tasks_cfg:
            print(f"❌ [{idx}/{len(shots)}] {shot_id} 未知的 task_type='{task_type}'，可用: {list(tasks_cfg.keys())}")
            fail_count += 1
            continue

        print(f"\n🎬 [{idx}/{len(shots)}] 开始处理镜头 {shot_id} | 类型: {task_type}")

        if args.dry_run:
            try:
                wf_payload, seed = build_prompt_payload(
                    shot,
                    tasks_cfg[task_type],
                    client=None,
                    media_dir=media_dir,
                    no_upload=True,
                )
                dry_dir = Path("workspace/logs/dry_run")
                dry_dir.mkdir(parents=True, exist_ok=True)
                out_json = dry_dir / f"{shot_id}_{task_type}.json"
                out_json.write_text(json.dumps(wf_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"    ✅ [dry-run] 已生成请求载荷 (seed={seed}) -> {out_json}")
                ok_count += 1
            except Exception as exc:
                print(f"    ❌ [dry-run] 参数映射失败: {exc}")
                fail_count += 1
            continue

        assert client is not None
        last_err: str | None = None
        for attempt in range(1, args.max_retries + 2):
            try:
                t0 = time.monotonic()
                wf_payload, seed = build_prompt_payload(
                    shot,
                    tasks_cfg[task_type],
                    client=client,
                    media_dir=media_dir,
                    no_upload=args.no_upload,
                )
                prompt_id = client.submit_prompt(wf_payload)
                print(f"    📨 已入队 (attempt {attempt}) | prompt_id={prompt_id} | seed={seed}", flush=True)

                hist = client.wait_for_completion(prompt_id, timeout_sec=args.timeout)
                saved = client.download_outputs(hist, output_dir / shot_id, shot_id, task_type)
                elapsed = round(time.monotonic() - t0, 1)

                state["shots"][shot_id] = {
                    "status": "done",
                    "task_type": task_type,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "elapsed_sec": elapsed,
                    "outputs": saved,
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                save_state(state_path, state)
                print(f"    ✅ 镜头 {shot_id} 渲染并回收完成 ({elapsed}s) -> {len(saved)} 个文件")
                for p in saved:
                    print(f"       📁 {p}")
                ok_count += 1
                last_err = None
                break
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                print(f"    ⚠️ 第 {attempt} 次尝试失败: {last_err}", flush=True)
                if attempt <= args.max_retries:
                    time.sleep(3)

        if last_err is not None:
            state["shots"][shot_id] = {
                "status": "failed",
                "task_type": task_type,
                "error": last_err,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            save_state(state_path, state)
            fail_count += 1

    print(f"\n📊 批量调度汇总: 成功 {ok_count} | 跳过 {skip_count} | 失败 {fail_count}")
    return 1 if fail_count > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
