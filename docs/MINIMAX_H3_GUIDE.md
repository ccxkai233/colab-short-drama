# 🎬 MiniMax H3 音画一体化短剧管线指南

本项目已全面接入 **ComfyUI 0.46.0+ 原生 MiniMax H3 音画一体化引擎**（零第三方插件依赖），可将过去「文生图 → 图生视频 → 外部配音 → 对口型」4 段流水线收拢为 1 次推理直出带对白、音效、配乐的成片。

---

## 一、模型与硬件配置（Colab L4 22GB / A100）

Colab Notebook Cell 3 默认勾选以下 MiniMax H3 套装（来自官方 `Comfy-Org/MiniMax-H3`，`cu130` 推荐 `int8_convrot`）：

| 模型组 | 文件 | 体积 | 用途 |
| :--- | :--- | :--- | :--- |
| `minimax_h3_core`（必选） | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB | Qwen3-VL-32B 文本/多模态编码器（**无需 Blackwell，L4/A100 可直接跑**） |
| `minimax_h3_core`（必选） | `minimax_h3_video_vae_int8_convrot.safetensors` + `minimax_h3_audio_vae_fp32.safetensors` | 3.2 GB | Video + Audio 双 VAE |
| `minimax_h3_fl2va` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GB | `h3_t2va`（纯文生音画）、`h3_i2va`（首帧/首尾帧音画） |
| `minimax_h3_ref2va` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB | `h3_r2v`（多图 + 视频 + 音色全参考） |
| `minimax_h3_turbo`（可选） | FL2V 8步 / REF2V 4步 Turbo LoRA | 3.6 GB | 低成本抽分镜草稿；定稿正式片建议关掉 Turbo 跑 `20~25` 步 |

---

## 二、MiniMax H3 硬规则（脚本已自动兜底）

1. **双轨模型不可混用**：
   - `h3_t2va` / `h3_i2va` → [workflows/h3_fl2va_api.json](../workflows/h3_fl2va_api.json)（`fl2va` 权重）
   - `h3_r2v` → [workflows/h3_r2v_api.json](../workflows/h3_r2v_api.json)（`ref2va` 权重）
2. **宽高与帧数对齐**：
   - 宽高必须为 **32 的倍数**（如 9:16 竖屏 `544×960` / `768×1344`，16:9 横屏 `960×544` / `1344×768`）；
   - 帧数（24fps）必须满足 **`17k + 5`**（`73`≈3s、`124`≈5s、`175`≈7.3s、`243`≈10s）。
   - `scripts/batch_runner.py` 已内置自动对齐：分镜表里直接写 `duration_sec: 5` 即可自动换算为 `124` 帧。
3. **参考视频必须 24fps**：
   - 运行 `python3 scripts/preprocess_media.py`（默认 `--profile h3`）会自动把参考视频重采样为 **24fps H.264 MP4**，把参考音频洗成 **32kHz PCM_16 WAV**。
   - 在 `h3_r2v` 里若希望同时参考原视频的声音，务必在分镜表加 `"ref_video_1_use_audio": true`（仅连 `ref_video_1` 模型听不到原视频声音）。
4. **无负面提示词（CFG = 1）**：
   - 全程使用 `BasicGuider`，**不要写 Negative Prompt**，也不要在正向描述里写“不要 xxx”。

---

## 三、结构化分镜脚本写法模板

```text
Keep the face, hairstyle, and outfit of <Picture 1>. Follow the timbre of <Audio 1> for speaker (S1).

integrated_multimodal_description:
[Shot 1] 00:00.000 - 00:02.500: 画面、景别、运镜与动作描述；画面内出现的招牌文字用英文双引号，如 neon sign reading "营业中"。
[Shot 2] At 00:02.500, cut to close-up of the woman speaking calmly, (S1) 这里直接写角色台词，不要加双引号。
overall_soundscape:
环境底噪与物理音效（雨声、脚步声、杯碟碰响）。
non_diegetic_music:
N/A
```

> ⚠️ **防乱加 BGM 坑点**：不需要配乐时，`non_diegetic_music:` 下务必明确写 **`N/A`**。

---

## 四、离线验证与执行命令

```bash
# 1. 预处理参考素材（默认 profile=h3 -> 视频 24fps，音频 32kHz WAV）
python3 scripts/preprocess_media.py --loudnorm

# 2. 离线演练 MiniMax H3 分镜表（验证 17k+5 帧数换算、32 倍数对齐、动态参考口连线与占位节点裁剪）
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json --dry-run

# 3. 正式提交云端渲染
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json
```
