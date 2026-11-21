#!/usr/bin/env python3
"""
Short-Drama-Factory 云端按需模型极速拉取脚本
特性：
  1. 强制开启 HF_HUB_ENABLE_HF_TRANSFER=1，Colab 内网满速直拉几十 GB 大模型到高速本地盘。
  2. 支持按模块分组勾选拉取（latentsync, wan22_fp8, wan22_fp16, flux_dev, flux_kontext, flux_schnell）。
  3. 自动补齐 ByteDance/LatentSync-1.6 缺失的离线 VAE（stabilityai/sd-vae-ft-mse -> models/checkpoints/LatentSync-1.6/vae/）。
  4. 自动扁平化 Comfy-Org 仓库的 split_files/ 路径，落盘到 ComfyUI 标准目录，不留脏目录。
  5. 幂等跳过已存在且非空的大模型文件，支持写入 Drive/logs/models_report.json。
"""

from __future__ import annotations

import os

# 必须在导入 huggingface_hub 之前设置，确保启用 Rust 高速传输引擎
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 内置兜底清单：即使只复制了 colab/ 目录、未复制 configs/models.yaml，依然可完整运行
FALLBACK_MANIFEST: dict[str, Any] = {
    "groups": {
        "latentsync": {
            "description": "ByteDance LatentSync-1.6 + 必需离线 VAE (stabilityai/sd-vae-ft-mse)",
            "items": [
                {
                    "name": "LatentSync-1.6 主权重与 Whisper",
                    "type": "snapshot",
                    "repo_id": "ByteDance/LatentSync-1.6",
                    "target_dir": "models/checkpoints/LatentSync-1.6",
                    "ignore_patterns": ["*.md", ".gitattributes"],
                },
                {
                    "name": "LatentSync 离线加载所需 VAE (sd-vae-ft-mse)",
                    "type": "snapshot",
                    "repo_id": "stabilityai/sd-vae-ft-mse",
                    "target_dir": "models/checkpoints/LatentSync-1.6/vae",
                    "ignore_patterns": ["*.md", ".gitattributes", "*.ckpt"],
                },
            ],
        },
        "wan22_fp8": {
            "description": "Wan 2.2 I2V 14B FP8 (L4/A100 通吃)",
            "items": [
                {
                    "name": "Wan 2.2 I2V 14B High-Noise FP8",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                    "target_dir": "models/diffusion_models",
                    "local_filename": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
                },
                {
                    "name": "Wan 2.2 I2V 14B Low-Noise FP8",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                    "target_dir": "models/diffusion_models",
                    "local_filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
                },
                {
                    "name": "Wan UMT5-XXL FP8 Text Encoder",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "target_dir": "models/text_encoders",
                    "local_filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                },
                {
                    "name": "Wan 2.1 VAE",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/vae/wan_2.1_vae.safetensors",
                    "target_dir": "models/vae",
                    "local_filename": "wan_2.1_vae.safetensors",
                },
            ],
        },
        "wan22_fp16": {
            "description": "Wan 2.2 I2V 14B FP16 原版（推荐 A100 80GB）",
            "items": [
                {
                    "name": "Wan 2.2 I2V 14B High-Noise FP16",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
                    "target_dir": "models/diffusion_models",
                    "local_filename": "wan2.2_i2v_high_noise_14B_fp16.safetensors",
                },
                {
                    "name": "Wan 2.2 I2V 14B Low-Noise FP16",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
                    "target_dir": "models/diffusion_models",
                    "local_filename": "wan2.2_i2v_low_noise_14B_fp16.safetensors",
                },
                {
                    "name": "Wan UMT5-XXL FP16 Text Encoder",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
                    "target_dir": "models/text_encoders",
                    "local_filename": "umt5_xxl_fp16.safetensors",
                },
                {
                    "name": "Wan 2.1 VAE",
                    "type": "file",
                    "repo_id": "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
                    "repo_filename": "split_files/vae/wan_2.1_vae.safetensors",
                    "target_dir": "models/vae",
                    "local_filename": "wan_2.1_vae.safetensors",
                },
            ],
        },
        "flux_dev": {
            "description": "FLUX.1-dev FP8 一体包 Checkpoint",
            "items": [
                {
                    "name": "FLUX.1-dev FP8 All-in-One Checkpoint",
                    "type": "file",
                    "repo_id": "Comfy-Org/flux1-dev",
                    "repo_filename": "flux1-dev-fp8.safetensors",
                    "target_dir": "models/checkpoints",
                    "local_filename": "flux1-dev-fp8.safetensors",
                },
            ],
        },
        "flux_kontext": {
            "description": "FLUX.1-Kontext-dev FP8 + CLIP/T5/VAE",
            "items": [
                {
                    "name": "FLUX.1-Kontext-dev FP8 Diffusion Model",
                    "type": "file",
                    "repo_id": "Comfy-Org/flux1-kontext-dev_ComfyUI",
                    "repo_filename": "split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
                    "target_dir": "models/diffusion_models",
                    "local_filename": "flux1-dev-kontext_fp8_scaled.safetensors",
                },
                {
                    "name": "FLUX CLIP-L Text Encoder",
                    "type": "file",
                    "repo_id": "comfyanonymous/flux_text_encoders",
                    "repo_filename": "clip_l.safetensors",
                    "target_dir": "models/text_encoders",
                    "local_filename": "clip_l.safetensors",
                },
                {
                    "name": "FLUX T5-XXL FP8 Text Encoder",
                    "type": "file",
                    "repo_id": "comfyanonymous/flux_text_encoders",
                    "repo_filename": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                    "target_dir": "models/text_encoders",
                    "local_filename": "t5xxl_fp8_e4m3fn_scaled.safetensors",
                },
                {
                    "name": "FLUX AE VAE",
                    "type": "file",
                    "repo_id": "black-forest-labs/FLUX.1-dev",
                    "repo_filename": "ae.safetensors",
                    "target_dir": "models/vae",
                    "local_filename": "ae.safetensors",
                },
            ],
        },
        "flux_schnell": {
            "description": "FLUX.1-schnell FP8 一体包",
            "items": [
                {
                    "name": "FLUX.1-schnell FP8 Checkpoint",
                    "type": "file",
                    "repo_id": "Comfy-Org/flux1-schnell",
                    "repo_filename": "flux1-schnell-fp8.safetensors",
                    "target_dir": "models/checkpoints",
                    "local_filename": "flux1-schnell-fp8.safetensors",
                },
            ],
        },
    }
}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[download_models {ts}] {msg}", flush=True)


def load_manifest(manifest_arg: str | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if manifest_arg:
        candidates.append(Path(manifest_arg))
    script_dir = Path(__file__).resolve().parent
    candidates.extend(
        [
            script_dir.parent / "configs" / "models.yaml",
            script_dir / "models.yaml",
            Path("/content/drive/MyDrive/ComfyUI_Work/scripts/models.yaml"),
        ]
    )
    for path in candidates:
        if path.exists():
            import yaml

            log(f"加载模型清单: {path}")
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    log("未找到外部 models.yaml，使用内置默认模型清单。")
    return FALLBACK_MANIFEST


def download_item(
    item: dict[str, Any],
    comfy_dir: Path,
    token: str | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, snapshot_download

    kind = item["type"]
    name = item.get("name", item.get("repo_id", "unnamed"))
    target_dir = comfy_dir / item["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    status = "downloaded"
    final_path: Path

    if kind == "snapshot":
        final_path = target_dir
        marker_exists = any(target_dir.iterdir())
        if marker_exists and not force:
            log(f"⏭️  跳过已存在目录（传入 --force 可强制校验）: {name} -> {target_dir}")
            status = "skipped_existing"
        else:
            log(f"⬇️  [snapshot] {name}: {item['repo_id']} -> {target_dir}")
            snapshot_download(
                repo_id=item["repo_id"],
                local_dir=str(target_dir),
                ignore_patterns=item.get("ignore_patterns"),
                token=token,
            )
    elif kind == "file":
        local_name = item.get("local_filename") or Path(item["repo_filename"]).name
        final_path = target_dir / local_name
        if final_path.exists() and final_path.stat().st_size > 0 and not force:
            size_gb = final_path.stat().st_size / (1024**3)
            log(f"⏭️  跳过已存在文件 ({size_gb:.2f} GB): {final_path.name}")
            status = "skipped_existing"
        else:
            log(f"⬇️  [file] {name}: {item['repo_id']}/{item['repo_filename']} -> {final_path}")
            cached_path = Path(
                hf_hub_download(
                    repo_id=item["repo_id"],
                    filename=item["repo_filename"],
                    token=token,
                )
            )
            # 优先硬链或软链到目标路径以节省一倍磁盘空间与复制时间（Colab 本地盘同分区下零拷贝）
            if final_path.exists() or final_path.is_symlink():
                final_path.unlink()
            try:
                os.link(cached_path.resolve(), final_path)
            except OSError:
                final_path.symlink_to(cached_path.resolve())
    elif kind == "url":
        local_name = item["local_filename"]
        final_path = target_dir / local_name
        if final_path.exists() and final_path.stat().st_size > 0 and not force:
            log(f"⏭️  跳过已存在直链文件: {final_path.name}")
            status = "skipped_existing"
        else:
            log(f"⬇️  [url] {name}: {item['url']} -> {final_path}")
            if shutil.which("aria2c"):
                subprocess.run(
                    [
                        "aria2c",
                        "-x",
                        "16",
                        "-s",
                        "16",
                        "-k",
                        "1M",
                        "-d",
                        str(target_dir),
                        "-o",
                        local_name,
                        item["url"],
                    ],
                    check=True,
                )
            else:
                subprocess.run(["curl", "-fSL", "-o", str(final_path), item["url"]], check=True)
    else:
        raise ValueError(f"不支持的下载类型: {kind}")

    elapsed = round(time.time() - t0, 1)
    log(f"✅ 完成: {name} ({status}, {elapsed}s)")
    return {
        "name": name,
        "type": kind,
        "target": str(final_path),
        "status": status,
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab ComfyUI 按需模型高速拉取器")
    parser.add_argument(
        "--groups",
        default="latentsync",
        help="逗号分隔的模型组名，例如: latentsync,wan22_fp8,flux_dev",
    )
    parser.add_argument("--all", action="store_true", help="下载清单中所有分组")
    parser.add_argument("--list", action="store_true", help="列出所有可用模型分组并退出")
    parser.add_argument("--comfy-dir", default="/content/ComfyUI", help="ComfyUI 本地根目录")
    parser.add_argument(
        "--drive-work-dir",
        default="/content/drive/MyDrive/ComfyUI_Work",
        help="Drive 工作目录（用于记录 models_report.json）",
    )
    parser.add_argument("--manifest", default=None, help="自定义 models.yaml 路径")
    parser.add_argument("--token", default=None, help="Hugging Face Token（默认读 HF_TOKEN 环境变量）")
    parser.add_argument("--force", action="store_true", help="强制重新校验/覆盖已有文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    groups_map: dict[str, Any] = manifest.get("groups", {})

    if args.list:
        print("可用模型分组：")
        for gname, gcfg in groups_map.items():
            print(f"  - {gname:14s} : {gcfg.get('description', '')}")
        return 0

    selected = list(groups_map.keys()) if args.all else [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = [g for g in selected if g not in groups_map]
    if unknown:
        raise SystemExit(f"未知的模型分组: {unknown}，可用分组: {list(groups_map.keys())}")

    token = args.token or os.environ.get("HF_TOKEN") or None
    comfy_dir = Path(args.comfy_dir).resolve()
    drive_work_dir = Path(args.drive_work_dir).resolve()

    log(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    log(f"即将拉取分组: {selected} | 目标 ComfyUI: {comfy_dir}")

    results: list[dict[str, Any]] = []
    for gname in selected:
        log(f"=== 分组 [{gname}]: {groups_map[gname].get('description', '')} ===")
        for item in groups_map[gname].get("items", []):
            results.append(download_item(item, comfy_dir, token, force=args.force))

    # 关键校验：如果选了 latentsync，确认离线 VAE 目录存在 config.json
    if "latentsync" in selected:
        vae_cfg = comfy_dir / "models" / "checkpoints" / "LatentSync-1.6" / "vae" / "config.json"
        if not vae_cfg.exists():
            log(f"⚠️ 警告：未检测到 LatentSync 离线 VAE 配置文件 {vae_cfg}，请检查下载日志！")
        else:
            log(f"✅ LatentSync-1.6 离线 VAE 校验通过: {vae_cfg}")

    if drive_work_dir.exists():
        log_dir = drive_work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / "models_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "selected_groups": selected,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"下载报告已保存: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
