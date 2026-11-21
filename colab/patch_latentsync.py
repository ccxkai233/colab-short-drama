#!/usr/bin/env python3
"""
comfyui-latentsync-enhanced 兼容性探测与自动热补丁脚本
职责：
  1. 在当前云端 PyTorch / torchaudio / torchvision 运行环境下做实测探测：
     - 探测 torchaudio.save 是否因后端/torchcodec 变更报错
     - 探测 torchvision.io.read_video 是否因后端移除或 torchcodec 缺失报错
  2. 若探测失败（或显式传入 --force），备份 nodes.py 为 nodes.py.bak，
     注入基于 soundfile + cv2 的安全替代函数，并替换 nodes.py 中的调用。
  3. 对补丁后的 nodes.py 执行 py_compile 语法校验；若失败则自动回滚备份。
  4. 写入诊断报告到 Drive/logs/patch_report.json。
"""

from __future__ import annotations

import argparse
import json
import py_compile
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PATCH_MARKER_START = "# --- SDF_COMPAT_PATCH_START ---"
PATCH_MARKER_END = "# --- SDF_COMPAT_PATCH_END ---"

COMPAT_HELPER_CODE = '''
# --- SDF_COMPAT_PATCH_START ---
import cv2 as _sdf_cv2
import numpy as _sdf_np
import soundfile as _sdf_sf
import subprocess as _sdf_subprocess
import tempfile as _sdf_tempfile
import os as _sdf_os
import torch as _sdf_torch


def _sdf_safe_torchaudio_save(uri, src, sample_rate, *args, **kwargs):
    """兼容替换 torchaudio.save：使用 soundfile 直接写出标准 WAV，避开新版 torchaudio/torchcodec 报错。"""
    arr = src.detach().cpu().float().numpy() if hasattr(src, "detach") else _sdf_np.asarray(src, dtype=_sdf_np.float32)
    # torchaudio 格式通常为 (channels, num_frames)，soundfile 需要 (num_frames, channels)
    if arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
        arr = arr.T
    arr = _sdf_np.clip(arr, -1.0, 1.0)
    _sdf_sf.write(str(uri), arr, int(sample_rate), subtype="PCM_16")


def _sdf_safe_read_video(filename, start_pts=0, end_pts=None, pts_unit="pts", output_format="THWC", **kwargs):
    """兼容替换 torchvision.io.read_video：使用 OpenCV (cv2) 读取帧并按原接口返回 (vframes, aframes, info)。"""
    cap = _sdf_cv2.VideoCapture(str(filename))
    if not cap.isOpened():
        raise RuntimeError(f"[SDF Patch] cv2 无法打开视频文件: {filename}")

    fps = float(cap.get(_sdf_cv2.CAP_PROP_FPS) or 25.0)
    frames = []
    idx = 0
    start_frame = int(round(float(start_pts) * fps)) if (pts_unit == "sec" and start_pts) else 0
    end_frame = int(round(float(end_pts) * fps)) if (pts_unit == "sec" and end_pts is not None) else None

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if idx >= start_frame and (end_frame is None or idx <= end_frame):
                rgb = _sdf_cv2.cvtColor(bgr, _sdf_cv2.COLOR_BGR2RGB)
                frames.append(rgb)
            idx += 1
    finally:
        cap.release()

    if frames:
        vframes = _sdf_torch.from_numpy(_sdf_np.stack(frames, axis=0)).to(_sdf_torch.uint8)
        if str(output_format).upper() == "TCHW":
            vframes = vframes.permute(0, 3, 1, 2).contiguous()
    else:
        vframes = _sdf_torch.empty((0, 0, 0, 3), dtype=_sdf_torch.uint8)

    # 尝试通过 ffmpeg 提取音轨（与 torchvision.io.read_video 保持返回三元组兼容）
    aframes = _sdf_torch.empty((0, 0), dtype=_sdf_torch.float32)
    audio_fps = None
    tmp_wav = None
    try:
        fd, tmp_wav = _sdf_tempfile.mkstemp(suffix=".wav")
        _sdf_os.close(fd)
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(filename),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            tmp_wav,
        ]
        res = _sdf_subprocess.run(cmd, stdout=_sdf_subprocess.DEVNULL, stderr=_sdf_subprocess.DEVNULL)
        if res.returncode == 0 and _sdf_os.path.getsize(tmp_wav) > 44:
            wav_data, sr = _sdf_sf.read(tmp_wav, dtype="float32", always_2d=True)
            aframes = _sdf_torch.from_numpy(wav_data.T.copy())
            audio_fps = int(sr)
    except Exception:
        pass
    finally:
        if tmp_wav and _sdf_os.path.exists(tmp_wav):
            try:
                _sdf_os.remove(tmp_wav)
            except OSError:
                pass

    info = {"video_fps": fps}
    if audio_fps is not None:
        info["audio_fps"] = audio_fps
    return vframes, aframes, info
# --- SDF_COMPAT_PATCH_END ---
'''.lstrip()


def log(msg: str) -> None:
    print(f"[patch_latentsync] {msg}", flush=True)


def find_latentsync_nodes(comfy_dir: Path) -> Path | None:
    custom_nodes = comfy_dir / "custom_nodes"
    if not custom_nodes.exists():
        return None
    for entry in custom_nodes.iterdir():
        if entry.is_dir() and "latentsync" in entry.name.lower():
            candidate = entry / "nodes.py"
            if candidate.exists():
                return candidate
    return None


def probe_torchaudio_save() -> tuple[bool, str]:
    """返回 (是否正常, 诊断说明)。"""
    try:
        import torch
        import torchaudio

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "probe.wav"
            dummy = torch.zeros((1, 1600), dtype=torch.float32)
            torchaudio.save(str(wav_path), dummy, 16000)
            if not wav_path.exists() or wav_path.stat().st_size == 0:
                return False, "torchaudio.save 生成空文件"
        return True, f"OK (torchaudio {getattr(torchaudio, '__version__', 'unknown')})"
    except Exception as exc:
        return False, f"FAILED: {type(exc).__name__}: {exc}"


def probe_torchvision_read_video() -> tuple[bool, str]:
    """返回 (是否正常, 诊断说明)。"""
    try:
        import cv2
        import numpy as np
        import torchvision

        read_video = getattr(getattr(torchvision, "io", None), "read_video", None)
        if read_video is None:
            return False, "torchvision.io.read_video 不存在"

        with tempfile.TemporaryDirectory() as tmpdir:
            mp4_path = Path(tmpdir) / "probe.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(mp4_path), fourcc, 25.0, (64, 64))
            for _ in range(4):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()

            vframes, _, _ = read_video(str(mp4_path), pts_unit="sec")
            if getattr(vframes, "shape", (0,))[0] == 0:
                return False, "torchvision.io.read_video 返回 0 帧"
        return True, f"OK (torchvision {getattr(torchvision, '__version__', 'unknown')})"
    except Exception as exc:
        return False, f"FAILED: {type(exc).__name__}: {exc}"


def apply_patch(
    nodes_path: Path,
    *,
    patch_audio: bool,
    patch_video: bool,
) -> dict:
    result = {
        "nodes_path": str(nodes_path),
        "patched_audio": False,
        "patched_video": False,
        "already_patched": False,
        "backup_path": str(nodes_path.with_suffix(".py.bak")),
    }

    content = nodes_path.read_text(encoding="utf-8")
    if PATCH_MARKER_START in content:
        log("检测到 nodes.py 已应用过 SDF 兼容补丁（幂等跳过注入）。")
        result["already_patched"] = True
        return result

    if not patch_audio and not patch_video:
        log("当前云端 PyTorch 探测全部通过，无需修改 nodes.py。")
        return result

    backup_path = nodes_path.with_suffix(".py.bak")
    if not backup_path.exists():
        shutil.copy2(nodes_path, backup_path)
        log(f"已备份原始文件 -> {backup_path}")

    new_content = content

    # 在顶部 __future__ 导入之后插入辅助函数（确保不破坏 from __future__ 规则）
    lines = new_content.splitlines(keepends=True)
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#!") or stripped.startswith("# -*-") or stripped.startswith("from __future__"):
            insert_idx = i + 1
        elif stripped and not stripped.startswith("#"):
            break
    lines.insert(insert_idx, "\n" + COMPAT_HELPER_CODE + "\n")
    new_content = "".join(lines)

    if patch_audio:
        replaced, count = re.subn(
            r"\btorchaudio\.save\s*\(",
            "_sdf_safe_torchaudio_save(",
            new_content,
        )
        new_content = replaced
        result["patched_audio"] = count > 0
        log(f"替换 torchaudio.save -> _sdf_safe_torchaudio_save（共 {count} 处）")

    if patch_video:
        replaced, count1 = re.subn(
            r"\btorchvision\.io\.read_video\s*\(",
            "_sdf_safe_read_video(",
            new_content,
        )
        # 若文件使用 `from torchvision.io import read_video` 形式，一并替换裸调用
        count2 = 0
        if re.search(r"from\s+torchvision\.io\s+import\s+.*\bread_video\b", replaced):
            replaced, count2 = re.subn(
                r"(?<!def )(?<!\.)\bread_video\s*\(",
                "_sdf_safe_read_video(",
                replaced,
            )
        new_content = replaced
        result["patched_video"] = (count1 + count2) > 0
        log(f"替换 torchvision.io.read_video -> _sdf_safe_read_video（共 {count1 + count2} 处）")

    nodes_path.write_text(new_content, encoding="utf-8")

    # 语法校验，失败则立即回滚
    try:
        py_compile.compile(str(nodes_path), doraise=True)
        log("✅ 补丁后 nodes.py 语法校验通过！")
    except Exception as exc:
        log(f"❌ 语法校验失败，正在自动回滚至备份: {exc}")
        shutil.copy2(backup_path, nodes_path)
        raise

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LatentSync-Enhanced 兼容性探测与自动补丁")
    parser.add_argument("--comfy-dir", default="/content/ComfyUI", help="ComfyUI 根目录")
    parser.add_argument(
        "--drive-work-dir",
        default="/content/drive/MyDrive/ComfyUI_Work",
        help="Drive 工作目录（用于写入 patch_report.json）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略探测结果，强制替换为 soundfile + cv2 实现",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅探测不修改任何文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comfy_dir = Path(args.comfy_dir).resolve()
    drive_work_dir = Path(args.drive_work_dir).resolve()

    nodes_path = find_latentsync_nodes(comfy_dir)
    if nodes_path is None:
        log(f"⚠️ 未在 {comfy_dir / 'custom_nodes'} 下找到 LatentSync 插件的 nodes.py，跳过补丁。")
        return 0

    audio_ok, audio_msg = probe_torchaudio_save()
    video_ok, video_msg = probe_torchvision_read_video()
    log(f"探测 torchaudio.save:          {audio_msg}")
    log(f"探测 torchvision.io.read_video: {video_msg}")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "probe": {
            "torchaudio_save_ok": audio_ok,
            "torchaudio_save_detail": audio_msg,
            "torchvision_read_video_ok": video_ok,
            "torchvision_read_video_detail": video_msg,
        },
        "force": args.force,
        "check_only": args.check_only,
        "patch_result": None,
    }

    if not args.check_only:
        patch_audio = args.force or (not audio_ok)
        patch_video = args.force or (not video_ok)
        report["patch_result"] = apply_patch(
            nodes_path,
            patch_audio=patch_audio,
            patch_video=patch_video,
        )

    log_dir = drive_work_dir / "logs"
    if drive_work_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)
        out = log_dir / "patch_report.json"
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"诊断报告已保存: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
