# 🎬 MiniMax H3 音画一体化短剧管线指南

本项目已全面接入 **ComfyUI 0.46.0+ 原生 MiniMax H3 音画一体化引擎**（零第三方插件依赖），可将过去「文生图 → 图生视频 → 外部配音 → 对口型」4 段流水线收拢为 1 次推理直出带对白、口型、环境音与配乐的成片。

---

## 一、模型与硬件配置（Colab L4 22GB / A100）

Colab Notebook Cell 3 默认模型组均来自官方 `Comfy-Org/MiniMax-H3`（你的 Colab 为 `PyTorch 2.14.1+cu130`，与官方推荐的 `int8_convrot` 量化权重完全匹配）：

| 模型组 | 文件 | 体积 | 用途与说明 |
| :--- | :--- | :--- | :--- |
| `minimax_h3_core`（必选） | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.6 GB | Qwen3-VL-32B 文本/多模态编码器（**无需 Blackwell，L4/A100 可直接跑**） |
| `minimax_h3_core`（必选） | `minimax_h3_video_vae_fp16.safetensors` + `minimax_h3_audio_vae_fp32.safetensors` | 5.4 GB | **原精度 `fp16` Video VAE** + `fp32` Audio VAE（细节锐度明显优于 int8 VAE） |
| `minimax_h3_fl2va` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.5 GB | `h3_t2va`（纯文生音画）、`h3_i2va`（首帧/首尾帧音画，画质与音质最稳） |
| `minimax_h3_ref2va` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 19.5 GB | `h3_r2v`（多图 + 视频 + 音色全参考，支持单段多镜头切镜） |
| `minimax_h3_turbo`（可选） | FL2V 8 步 + REF2V 4 步 Turbo LoRA | 3.6 GB | 快速抽分镜草稿（落盘到 `Drive/loras/`，下载一次永久复用）；定稿出正片建议关闭 |
| `upscale_2x`（默认勾选） | `RealESRGAN_x2plus.pth` | 64 MB | `upscale_video` 2x 高清视频超分，原音轨与帧率无损透传 |

> 💡 **快速开机小技巧**：如果本次只用首帧图拍 `h3_i2va`，Cell 3 只需勾 `MINIMAX_H3_CORE` + `MINIMAX_H3_FL2VA`（约 39.5 GB），关掉 `REF2VA` 与 `FLUX_DEV` 可省下 30+ GB 下载时间。

---

## 二、Colab NVIDIA L4（22GB）实测耗时与画质基准

基于本项目在 Colab L4 上已完整跑通的实测数据：

| 规格 | 帧数 / 时长 | 步数 & VAE | L4 实测耗时 | 文件大小 | 适用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `960×544`（横屏 16:9，双人中景） | `243` 帧（10.125s） | 20 步 + int8 VAE | **20.3 分钟**（1220s） | 3.0 MB | 低成本中景试戏、验证长段双人对白节奏 |
| `768×1344`（竖屏 9:16 满血，单人近景） | `192` 帧（8.0s） | 25 步 + **fp16 VAE** | **50.1 分钟**（3007s） | 4.6 MB | **正式成片级清晰度**（睫毛/发丝/刺绣锐利，口型精准） |
| `768×1344` + `h3_i2va_turbo`（预估） | `124~192` 帧（5~8s） | 8 步 Turbo LoRA | **~8~16 分钟** | — | 高清构图快速试戏、排查台词情绪与表情踩雷 |

> ⏱️ `.env` 默认 `JOB_TIMEOUT_SECONDS=3600`（60 分钟），确保 L4 跑 8~10 秒满血 768p 不会误超时。

---

## 三、MiniMax H3 硬规则（脚本已自动兜底）

1. **双轨模型不可混用**：
   - `h3_t2va` / `h3_i2va` → [workflows/h3_fl2va_api.json](../workflows/h3_fl2va_api.json)（`fl2va` 权重）
   - `h3_r2v` → [workflows/h3_r2v_api.json](../workflows/h3_r2v_api.json)（`ref2va` 权重）
2. **宽高与帧数对齐**：
   - 宽高必须为 **32 的倍数**（竖屏 9:16 推荐 `544×960` 试戏 / `768×1344` 满血；横屏 16:9 推荐 `960×544` 试戏 / `1344×768` 满血）；
   - 帧数（24fps）必须满足 **`17k + 5`**（`73`≈3s、`124`≈5.17s、`175`≈7.29s、`192`=8.0s、`243`≈10.125s）。
   - `scripts/batch_runner.py` 已内置自动对齐：分镜表直接写 `duration_sec: 8` 会自动换算为 `192` 帧。
3. **参考视频必须 24fps**：
   - 运行 `python3 scripts/preprocess_media.py`（默认 `--profile h3`）会自动把参考视频重采样为 **24fps H.264 MP4**，把参考音频洗成 **32kHz PCM_16 WAV**。
   - 在 `h3_r2v` 里若希望同时参考原视频声音，务必在分镜表加 `"ref_video_1_use_audio": true`（仅连 `ref_video_1` 模型听不到原视频声音）。
4. **无负面提示词（CFG = 1）**：
   - 全程使用 `BasicGuider`，**不要写 Negative Prompt**，也不要在正向描述里写“不要 xxx”。

---

## 四、结构化分镜脚本写法与实测避坑

```text
Keep the face, hairstyle, and outfit of <Picture 1>. Follow the timbre of <Audio 1> for speaker (S1).

integrated_multimodal_description:
[Shot 1] 00:00.000 - 00:02.500: 画面、景别、运镜与动作描述；画面内出现的招牌文字用英文双引号，如 neon sign reading "营业中"。
[Shot 2] At 00:02.500, cut to close-up of the woman speaking calmly, (S1) 这里直接写角色台词，不要加双引号。
overall_soundscape:
环境底噪与物理音效（竹叶风声、庭院远鸟、衣袖微响）。
non_diegetic_music:
N/A
```

### 实测避坑笔记
- **防乱加 BGM**：不需要配乐时，`non_diegetic_music:` 下必须明确写 **`N/A`**。
- **防“苦笑”笑得太甜**：实测在英文提示词里写 `bitter smile` 时，模型容易演成露齿甜笑；若要克制委屈/冷笑，建议明确写 **`a cold, faint closed-mouth smirk with reddened eyes, lips pressed slightly`**。
- **双人对白 vs 单人近景**：中景双人同框单张脸只占画面 ~8%，尽量将重要情绪对白拆成 **单人近景/特写正反打**，面部清晰度与演技感染力直接提升 3 倍以上。
- **试戏与重抽预期**：换分辨率或开关 Turbo LoRA 重跑时，即便同一 `seed` 演技细节也会重新生成；试戏主要用来验证**剧本/情绪指令/停顿是否准确**。如果某条低清试戏刚好演出了神级表情和台词，不要重抽，直接喂给 `upscale_video` 2x 超分保住原演技与原音轨。

---

## 五、多实例并发加速、Turbo 试戏与 2x 视频超分

### 1. 多台 Colab L4 并行抢单（总算力不变，等待时间除以 N）
1. 在浏览器开 **2~3 个 Colab 标签页**，各打开最新 Notebook 运行（Cell 4 已为每台分配独立 `worker_id`，多台共用同一 Google Drive 不会覆盖日志）。
2. 把各台打印出的 `trycloudflare.com` 地址用英文逗号拼进 `.env` 的 `COLAB_API_URLS`（或命令行直接传 `--urls`）：
   ```bash
   # 一键并发体检全部实例（自动汇总在线数与总显存）
   python3 scripts/check_cloud.py

   # 多实例并发抢单渲染（某台断线/530 时自动摘除并把镜头退回队列由存活节点接管）
   python3 scripts/batch_runner.py --storyboard workspace/storyboards/ep01_scene01_h3.json
   ```

### 2. Turbo 8 步 / 4 步草稿试戏模式
- 在 Colab Cell 3 勾选 **`DOWNLOAD_MINIMAX_H3_TURBO`**（首次落盘到 `Drive/loras/`，之后永久持久化无需重下）。
- 分镜表中把 `task_type` 设为 `h3_i2va_turbo` / `h3_t2va_turbo`（8 步）或 `h3_r2v_turbo`（4 步），耗时降至约 1/3；定稿出正片时改回 `h3_i2va` / `h3_r2v` 跑 20~25 步。

### 3. 2x 视频高清超分（保留原演技与原音轨）
- Cell 3 默认勾选 **`DOWNLOAD_UPSCALE_2X`**（`RealESRGAN_x2plus.pth`，仅 64 MB）。
- 在分镜表建一条 `task_type: "upscale_video"` 并填 `"video": "路径/原片.mp4"` 即可 2 倍超分（`960×544 → 1920×1088`，`768×1344 → 1536×2688`），原音轨与帧率 100% 无损保留。
