#!/usr/bin/env python3
"""
Short-Drama-Factory 本地素材批处理预处理工具（零第三方 Python 依赖，仅需 ffmpeg/ffprobe）
职责：
  1. 音频洗帧标准化：批量转为 16kHz、单声道、16-bit PCM .wav，自动跳过坏帧（-fflags +discardcorrupt -err_detect ignore_err），
     可选开启 EBU R128 响度归一化（--loudnorm）。
  2. 视频帧率统一对齐：批量转为标准 H.264 yuv420p .mp4，默认对齐 25fps（LatentSync-1.6 标准，可通过 --fps 自定义如 16/24）。
  3. 图片安全拷贝与命名标准化：分镜底图统一清理中文/空格/特殊字符文件名。
  4. 文件名清洗与映射留档：生成 clean_input/rename_manifest.json，完整记录「原文件名 <-> 清洗后文件名」及媒体元信息，
     并采用确定性短哈希防止中文名清洗后重名碰撞。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def check_ffmpeg() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(
            f"❌ 未检测到系统工具: {', '.join(missing)}。\n"
            "请先安装 ffmpeg（Ubuntu/Debian: sudo apt install ffmpeg | macOS: brew install ffmpeg | Windows: winget install Gyan.FFmpeg）。"
        )


def sanitize_stem(original_rel_path: str, stem: str) -> str:
    """将含中文、空格或特殊字符的文件名清洗为安全 ASCII 名称，并在发生丢字时附加确定性短哈希防冲突。"""
    normalized = unicodedata.normalize("NFKD", stem)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_only).strip("_-").lower()

    had_non_ascii_or_changed = cleaned != stem.lower()
    short_hash = hashlib.sha1(original_rel_path.encode("utf-8")).hexdigest()[:6]

    if not cleaned:
        return f"media_{short_hash}"
    if had_non_ascii_or_changed:
        # 若原文件名包含中文或特殊符号，附带短哈希确保「第一场_台词1」和「第一场_台词2」绝不重名
        return f"{cleaned}_{short_hash}"
    return cleaned


def probe_media(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not res.stdout.strip():
        return {}
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError:
        return {}

    info: dict[str, Any] = {}
    fmt = data.get("format", {})
    if "duration" in fmt:
        try:
            info["duration_sec"] = round(float(fmt["duration"]), 3)
        except ValueError:
            pass

    for stream in data.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and "width" not in info:
            info["width"] = stream.get("width")
            info["height"] = stream.get("height")
            info["video_codec"] = stream.get("codec_name")
            r_fps = stream.get("r_frame_rate", "")
            if "/" in r_fps:
                num, den = r_fps.split("/", 1)
                try:
                    if float(den) != 0:
                        info["fps"] = round(float(num) / float(den), 3)
                except ValueError:
                    pass
        elif ctype == "audio" and "sample_rate" not in info:
            info["audio_codec"] = stream.get("codec_name")
            try:
                info["sample_rate"] = int(stream.get("sample_rate", 0))
            except ValueError:
                pass
            info["channels"] = stream.get("channels")
    return info


def process_audio(
    src: Path,
    dst: Path,
    *,
    loudnorm: bool = False,
) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        str(src),
        "-vn",
    ]
    if loudnorm:
        cmd.extend(["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    cmd.extend(
        [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return False, res.stderr.strip() or f"ffmpeg exit code {res.returncode}"
    return True, "ok"


def process_video(
    src: Path,
    dst: Path,
    *,
    fps: float = 25.0,
    max_side: int | None = None,
    crf: int = 18,
    strip_audio: bool = False,
) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)

    vf_parts = [f"fps={fps:g}"]
    if max_side and max_side > 0:
        # 按长边等比缩放（不放大），且确保宽高为偶数满足 H.264 yuv420p 要求
        vf_parts.append(
            f"scale='if(gt(iw,ih),min(iw,{max_side}),-2)':'if(gt(iw,ih),-2,min(ih,{max_side}))'"
        )
    vf_parts.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    vf_expr = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        str(src),
        "-map",
        "0:v:0",
    ]
    if not strip_audio:
        # 0:a:0? 中的问号表示：若原视频无音轨则自动忽略，不报错
        cmd.extend(["-map", "0:a:0?"])
    cmd.extend(
        [
            "-vf",
            vf_expr,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
            "-preset",
            "fast",
        ]
    )
    if strip_audio:
        cmd.append("-an")
    else:
        cmd.extend(["-c:a", "aac", "-ar", "16000", "-ac", "1", "-b:a", "128k"])
    cmd.extend(["-movflags", "+faststart", str(dst)])

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return False, res.stderr.strip() or f"ffmpeg exit code {res.returncode}"
    return True, "ok"


def process_image(src: Path, dst: Path) -> tuple[bool, str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True, "ok"


def classify_file(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return None


def handle_single_file(
    src: Path,
    input_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    rel = src.relative_to(input_root).as_posix()
    kind = classify_file(src)
    assert kind is not None

    clean_stem = sanitize_stem(rel, src.stem)
    if kind == "audio":
        clean_name = f"{clean_stem}.wav"
        sub_folder = "audio"
    elif kind == "video":
        clean_name = f"{clean_stem}.mp4"
        sub_folder = "video"
    else:
        clean_name = f"{clean_stem}{src.suffix.lower()}"
        sub_folder = "images"

    dst = output_root / sub_folder / clean_name
    record: dict[str, Any] = {
        "kind": kind,
        "original_rel": rel,
        "clean_rel": dst.relative_to(output_root).as_posix(),
        "clean_filename": clean_name,
        "status": "pending",
        "message": "",
        "meta": {},
    }

    if args.dry_run:
        record["status"] = "dry_run"
        return record

    if dst.exists() and not args.overwrite:
        record["status"] = "skipped_existing"
        record["meta"] = probe_media(dst) if kind in ("audio", "video") else {}
        return record

    if kind == "audio":
        ok, msg = process_audio(src, dst, loudnorm=args.loudnorm)
    elif kind == "video":
        ok, msg = process_video(
            src,
            dst,
            fps=args.fps,
            max_side=args.max_side,
            crf=args.crf,
            strip_audio=args.strip_audio,
        )
    else:
        ok, msg = process_image(src, dst)

    record["status"] = "ok" if ok else "error"
    record["message"] = msg
    if ok and kind in ("audio", "video"):
        record["meta"] = probe_media(dst)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI 短剧本地素材预处理：16kHz 单声道 WAV 洗帧 + 视频帧率对齐 + 文件名安全清洗"
    )
    parser.add_argument(
        "--input-dir",
        default="workspace/raw_input",
        help="原始素材目录（递归扫描，默认 workspace/raw_input）",
    )
    parser.add_argument(
        "--output-dir",
        default="workspace/clean_input",
        help="清洗后输出目录（默认 workspace/clean_input）",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "audio", "video", "image"],
        default="all",
        help="处理的素材类型（默认 all）",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="视频统一目标帧率（LatentSync-1.6 默认 25，Wan 2.2 可选 16/24）",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="可选：限制视频长边最大像素（如 1080 或 1280，不填则保留原分辨率）",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频 H.264 压缩质量（默认 18，越小越无损）")
    parser.add_argument("--loudnorm", action="store_true", help="对干声音频开启 EBU R128 响度归一化")
    parser.add_argument("--strip-audio", action="store_true", help="视频预处理时移除原音轨（纯画面驱动场景）")
    parser.add_argument("--workers", type=int, default=4, help="并发处理线程数（默认 4）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")
    parser.add_argument("--dry-run", action="store_true", help="仅预览文件清洗与重命名计划，不执行转码")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        check_ffmpeg()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not input_root.exists():
        raise SystemExit(f"❌ 输入目录不存在: {input_root}")

    candidates: list[Path] = []
    for path in sorted(input_root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        kind = classify_file(path)
        if kind is None:
            continue
        if args.mode != "all" and kind != args.mode:
            continue
        candidates.append(path)

    if not candidates:
        print(f"⚠️ 在 {input_root} 中未找到匹配模式 [{args.mode}] 的素材文件。")
        return 0

    print(
        f"🎬 发现 {len(candidates)} 个素材文件 | 模式: {args.mode} | 目标帧率: {args.fps:g}fps | "
        f"输出: {output_root} {'[DRY-RUN]' if args.dry_run else ''}"
    )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(handle_single_file, src, input_root, output_root, args): src
            for src in candidates
        }
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            icon = {"ok": "✅", "skipped_existing": "⏭️", "dry_run": "🔍", "error": "❌"}.get(
                rec["status"], "•"
            )
            detail = f" ({rec['message']})" if rec["status"] == "error" else ""
            print(f"  {icon} [{rec['kind']:5s}] {rec['original_rel']} -> {rec['clean_rel']}{detail}")

    records.sort(key=lambda r: r["original_rel"])
    ok_cnt = sum(1 for r in records if r["status"] == "ok")
    skip_cnt = sum(1 for r in records if r["status"] == "skipped_existing")
    err_cnt = sum(1 for r in records if r["status"] == "error")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "rename_manifest.json"
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_dir": str(input_root),
            "output_dir": str(output_root),
            "video_target_fps": args.fps,
            "audio_format": "16kHz mono PCM_16 WAV",
            "summary": {"total": len(records), "ok": ok_cnt, "skipped": skip_cnt, "errors": err_cnt},
            "files": records,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📄 映射清单已写入: {manifest_path}")

    print(f"完成统计: 成功 {ok_cnt} | 跳过 {skip_cnt} | 失败 {err_cnt}")
    return 1 if err_cnt > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
