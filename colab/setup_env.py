#!/usr/bin/env python3
"""
Short-Drama-Factory 云端环境一键部署脚本（Google Colab 专用）
职责：
  1. 建立动静分离目录结构：Drive 持久化（input/output/workflows/loras/scripts/logs），
     Colab 高速本地盘存放 ComfyUI 本体与大模型。
  2. 安装系统依赖（ffmpeg, aria2, cloudflared）与 uv。
  3. 克隆 ComfyUI 与三大核心插件：
     - ComfyUI-Manager
     - ComfyUI-Inpaint-CropAndStitch
     - GuardSkill/ComfyUI-LatentSyncEnhanced
  4. 使用 uv 极速安装依赖，并显式补齐 comfyui-latentsync-enhanced 缺失的依赖包，
     同时保护 Colab 预装的 CUDA PyTorch 不被意外覆盖。
  5. 输出环境体检快照到 Drive/logs/env_report.json，便于远程排障。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CORE_PLUGINS = [
    (
        "ComfyUI-Manager",
        "https://github.com/ltdrdata/ComfyUI-Manager.git",
    ),
    (
        "ComfyUI-Inpaint-CropAndStitch",
        "https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch.git",
    ),
    (
        "ComfyUI-LatentSyncEnhanced",
        "https://github.com/GuardSkill/ComfyUI-LatentSyncEnhanced.git",
    ),
]

# comfyui-latentsync-enhanced 已知缺失依赖（显式补装）+ 工具依赖
EXTRA_PIP_PACKAGES = [
    "insightface",
    "onnxruntime-gpu",
    "librosa",
    "DeepCache",
    "imageio",
    "imageio-ffmpeg",
    "soundfile",
    "matplotlib",
    "more-itertools",
    "ffmpeg-python",
    "opencv-python-headless",
    "huggingface_hub",
    "hf_transfer",
    "pyyaml",
    "requests",
]

# 避免覆盖 Colab 自带的 CUDA 版 PyTorch 生态包
PROTECTED_TORCH_PREFIXES = (
    "torch",
    "torchvision",
    "torchaudio",
    "xformers",
)

DRIVE_SUBDIRS = ("input", "output", "workflows", "loras", "scripts", "logs")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[setup_env {ts}] {msg}", flush=True)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=True,
        env=env,
    )


def pick_comfy_dir(requested: str | None) -> Path:
    """选择 ComfyUI 存放路径（默认 /content/ComfyUI，优先本地高速盘）。"""
    if requested:
        return Path(requested).resolve()
    candidates = [Path("/content"), Path("/tmp")]
    for base in candidates:
        if base.exists() and os.access(base, os.W_OK):
            return base / "ComfyUI"
    return Path.cwd() / "ComfyUI"


def ensure_drive_dirs(drive_work_dir: Path) -> None:
    log(f"初始化 Google Drive 持久化目录: {drive_work_dir}")
    drive_work_dir.mkdir(parents=True, exist_ok=True)
    for sub in DRIVE_SUBDIRS:
        (drive_work_dir / sub).mkdir(parents=True, exist_ok=True)


def ensure_system_tools() -> None:
    """安装 ffmpeg、aria2 与 cloudflared。"""
    missing_apt = []
    if shutil.which("ffmpeg") is None:
        missing_apt.append("ffmpeg")
    if shutil.which("aria2c") is None:
        missing_apt.append("aria2")

    if missing_apt and shutil.which("apt-get"):
        log(f"安装系统依赖: {', '.join(missing_apt)}")
        run_cmd(["apt-get", "update", "-qq"], check=False)
        run_cmd(["apt-get", "install", "-y", "-qq", *missing_apt])
    else:
        log("ffmpeg / aria2 已就绪（或非 apt 环境，跳过）")

    if shutil.which("cloudflared") is None:
        target = Path("/usr/local/bin/cloudflared")
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        log(f"下载 cloudflared -> {target}")
        if shutil.which("aria2c"):
            run_cmd(
                [
                    "aria2c",
                    "-x",
                    "8",
                    "-s",
                    "8",
                    "-d",
                    str(target.parent),
                    "-o",
                    target.name,
                    url,
                ]
            )
        else:
            run_cmd(["curl", "-fsSL", "-o", str(target), url])
        target.chmod(0o755)
    else:
        log(f"cloudflared 已就绪: {shutil.which('cloudflared')}")


def ensure_uv() -> str:
    uv_bin = shutil.which("uv")
    if uv_bin:
        log(f"uv 已就绪: {uv_bin}")
        return uv_bin
    log("安装 uv 极速包管理器...")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv_bin = shutil.which("uv")
    if not uv_bin:
        raise RuntimeError("uv 安装后未在 PATH 中找到，请检查环境。")
    return uv_bin


def uv_pip_install(uv_bin: str, args: Iterable[str], *, check: bool = True) -> bool:
    cmd = [uv_bin, "pip", "install", "--system", "--break-system-packages", "--quiet", *args]
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        if check:
            raise subprocess.CalledProcessError(res.returncode, cmd)
        log(f"⚠️ 部分依赖安装返回非零状态（已容错继续）: {' '.join(args)}")
        return False
    return True


def filter_requirements_file(req_path: Path, out_path: Path) -> bool:
    """过滤掉可能覆盖 Colab 预装 CUDA PyTorch 的行，返回是否存在有效依赖行。"""
    lines: list[str] = []
    for raw in req_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pkg_head = (
            stripped.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split("!=")[0]
            .split("[")[0]
            .strip()
            .lower()
        )
        if pkg_head in PROTECTED_TORCH_PREFIXES:
            log(f"  跳过受保护包（保留 Colab CUDA PyTorch）: {stripped}")
            continue
        lines.append(stripped)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return bool(lines)


def clone_or_update(repo_url: str, target_dir: Path, *, update: bool = False) -> None:
    if (target_dir / ".git").exists():
        log(f"仓库已存在: {target_dir.name}")
        if update:
            run_cmd(["git", "pull", "--ff-only"], cwd=target_dir, check=False)
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(["git", "clone", "--depth", "1", repo_url, str(target_dir)])


def safe_symlink(link_path: Path, target_path: Path) -> None:
    """将 ComfyUI 内的持久化目录软链到 Google Drive（幂等安全）。"""
    target_path.mkdir(parents=True, exist_ok=True)
    link_path.parent.mkdir(parents=True, exist_ok=True)

    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            log(f"软链已就绪: {link_path} -> {target_path}")
            return
        link_path.unlink()
    elif link_path.exists():
        # 如果原本是 ComfyUI 初始化出来的空/示例文件夹，先迁移未冲突文件再替换
        if link_path.is_dir():
            for child in link_path.iterdir():
                dest = target_path / child.name
                if not dest.exists():
                    shutil.move(str(child), str(dest))
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    link_path.symlink_to(target_path, target_is_directory=True)
    log(f"已建立软链: {link_path} -> {target_path}")


def setup_symlinks(comfy_dir: Path, drive_work_dir: Path) -> None:
    log("配置动静分离软链接（Drive 仅存 input / output / loras / workflows）...")
    safe_symlink(comfy_dir / "input", drive_work_dir / "input")
    safe_symlink(comfy_dir / "output", drive_work_dir / "output")
    safe_symlink(comfy_dir / "models" / "loras", drive_work_dir / "loras")
    safe_symlink(
        comfy_dir / "user" / "default" / "workflows",
        drive_work_dir / "workflows",
    )


def collect_env_report(comfy_dir: Path, drive_work_dir: Path) -> dict:
    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "comfy_dir": str(comfy_dir),
        "drive_work_dir": str(drive_work_dir),
        "disk_free_gb": {},
        "gpu": "unknown",
        "packages": {},
        "custom_nodes": [],
    }

    for label, p in [("comfy_disk", comfy_dir), ("drive_disk", drive_work_dir)]:
        try:
            usage = shutil.disk_usage(p)
            report["disk_free_gb"][label] = round(usage.free / (1024**3), 2)
        except OSError:
            pass

    if shutil.which("nvidia-smi"):
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            capture_output=True,
        )
        if res.returncode == 0:
            report["gpu"] = res.stdout.strip()

    for mod in (
        "torch",
        "torchvision",
        "torchaudio",
        "onnxruntime",
        "insightface",
        "librosa",
        "soundfile",
        "cv2",
        "DeepCache",
        "huggingface_hub",
        "hf_transfer",
    ):
        try:
            m = __import__(mod)
            report["packages"][mod] = getattr(m, "__version__", "installed")
        except Exception as exc:
            report["packages"][mod] = f"MISSING ({type(exc).__name__}: {exc})"

    custom_nodes_dir = comfy_dir / "custom_nodes"
    if custom_nodes_dir.exists():
        for entry in sorted(custom_nodes_dir.iterdir()):
            if entry.is_dir() and not entry.name.startswith("."):
                commit = ""
                if (entry / ".git").exists():
                    r = subprocess.run(
                        ["git", "rev-parse", "--short", "HEAD"],
                        cwd=str(entry),
                        text=True,
                        capture_output=True,
                    )
                    if r.returncode == 0:
                        commit = r.stdout.strip()
                report["custom_nodes"].append({"name": entry.name, "commit": commit})

    out_file = drive_work_dir / "logs" / "env_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入环境体检快照: {out_file}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab ComfyUI 云端环境部署脚本")
    parser.add_argument(
        "--drive-work-dir",
        default="/content/drive/MyDrive/ComfyUI_Work",
        help="Google Drive 持久化工作目录",
    )
    parser.add_argument(
        "--comfy-dir",
        default=None,
        help="Colab 本地高速盘 ComfyUI 安装目录（默认 /content/ComfyUI）",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="若仓库已存在则执行 git pull 更新",
    )
    parser.add_argument(
        "--skip-system-tools",
        action="store_true",
        help="跳过 apt/cloudflared 安装（调试用）",
    )
    return parser.parse_args()


def main() -> int:
    start = time.time()
    args = parse_args()
    drive_work_dir = Path(args.drive_work_dir).resolve()
    comfy_dir = pick_comfy_dir(args.comfy_dir)

    ensure_drive_dirs(drive_work_dir)
    if not args.skip_system_tools:
        ensure_system_tools()
    uv_bin = ensure_uv()

    # 1. 部署 ComfyUI 本体
    log(f"部署 ComfyUI 本体 -> {comfy_dir}")
    clone_or_update(
        "https://github.com/comfyanonymous/ComfyUI.git",
        comfy_dir,
        update=args.update,
    )
    comfy_req = comfy_dir / "requirements.txt"
    if comfy_req.exists():
        filtered = Path("/tmp/comfyui_req_filtered.txt")
        if filter_requirements_file(comfy_req, filtered):
            uv_pip_install(uv_bin, ["-r", str(filtered)])

    # 2. 部署三大核心插件
    custom_nodes_dir = comfy_dir / "custom_nodes"
    for folder_name, repo_url in CORE_PLUGINS:
        plugin_dir = custom_nodes_dir / folder_name
        log(f"安装插件: {folder_name}")
        clone_or_update(repo_url, plugin_dir, update=args.update)
        plugin_req = plugin_dir / "requirements.txt"
        if plugin_req.exists():
            filtered = Path(f"/tmp/req_{folder_name}.txt")
            if filter_requirements_file(plugin_req, filtered):
                uv_pip_install(uv_bin, ["-r", str(filtered)], check=False)

    # 3. 显式补齐 LatentSync-Enhanced 缺失依赖 + 工具库
    log("显式补装 LatentSync-Enhanced 缺失依赖与高速下载组件...")
    uv_pip_install(uv_bin, EXTRA_PIP_PACKAGES, check=False)

    # 4. 建立动静分离软链
    setup_symlinks(comfy_dir, drive_work_dir)

    # 5. 生成环境诊断快照
    report = collect_env_report(comfy_dir, drive_work_dir)
    elapsed = round(time.time() - start, 1)
    log(f"✅ 云端环境部署完成！耗时 {elapsed}s | GPU: {report.get('gpu')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
