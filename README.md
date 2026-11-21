# 🎬 Short-Drama-Factory（本地大脑 + 云端肌肉 AI 短剧自动化生产管线）

一套面向 **本地 CLI 调度 + Google Colab Pro+（单机或多台 L4 / A100 并行）ComfyUI 渲染** 的轻量化 AI 短剧自动化生产管线，现已全面升级为 **ComfyUI 原生 MiniMax H3 音画一体化主力架构**（同时保留 FLUX / Wan 2.2 / IndexTTS2 / LatentSync 作为备用链路）。

---

## 核心架构特点

0. **主力管线：MiniMax H3 音画一体化直出（零第三方插件依赖）**
   - 直接调用 ComfyUI `0.46.0+` 原生 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo` / `VAEDecodeAudio` 节点，**一次推理同步输出画面 + 中文对白 + 口型 + 环境音效**，不再强依赖“外部配音 + 单独对口型 + 角色 LoRA”长链条。
   - **双轨模型**：`fl2va`（文生音画 `h3_t2va` / 首尾帧 `h3_i2va`）与 `ref2va`（多图/视频/音色全参考 `h3_r2v`），搭配 **原精度 `fp16` Video VAE** 与 **NVFP4 `Qwen3-VL-32B` 文本编码器**（Colab L4 22GB / A100 直接运行）。
   - **提速与高清闭环**：支持 **Turbo 8 步 / 4 步快速试戏**（`h3_i2va_turbo` / `h3_r2v_turbo`）、**2x 原生视频超分**（`upscale_video`，保留原音轨与帧率）以及 **多台 Colab L4 并发抢单池**。
   - *传统备用链路*：依然完整保留 `FLUX.1-dev FP8`（出角色定妆照/三视图）、`Wan 2.2 14B FP8`、`LatentSync-1.6` 以及本地 Win11 `IndexTTS2` 配音手册。

1. **动静分离 + 多实例日志隔离**
   - **Google Drive（`MyDrive/ComfyUI_Work/`）**：持久化 `input/`、`output/`、`workflows/`、`loras/`、`scripts/`、`logs/`；多台 Colab 实例同时挂载同一 Drive 时按独立 `worker_id` 分离日志（`comfyui_<worker_id>.log`、`tunnel_<worker_id>.txt`、`active_tunnels.txt`），互不覆盖。
   - **Colab 高速本地盘（`/content/ComfyUI`）**：存放 ComfyUI 本体与基础大模型，开机通过 `HF_HUB_ENABLE_HF_TRANSFER=1` 满速直拉，关机自动清空；内置 `colab/idle_unassign.py` 空闲看门狗，队列清空满 15 分钟自动 `unassign()` 停表省算力。

2. **本地大脑 CLI 工具链**
   - [scripts/preprocess_media.py](scripts/preprocess_media.py)：默认 `--profile h3` 批量把参考视频洗成 **24fps H.264 MP4**（防 H3 硬读 24fps 变慢动作）、参考音频洗成 **32kHz PCM_16 WAV**（匹配 H3 Audio VAE），自动清洗中文/空格文件名；老链路可用 `--profile latentsync` 切回 25fps / 16kHz。
   - [scripts/check_cloud.py](scripts/check_cloud.py)：支持**单实例或多实例并发远程体检**（GPU 显存、MiniMax H3 原生节点、模型目录、队列状态、最近执行报错），并支持一键远程中断/清队列。
   - [scripts/batch_runner.py](scripts/batch_runner.py)：读取 CSV/JSON 分镜表，自动把 `duration_sec` 换算对齐为 H3 的 `17k + 5` 帧、宽高对齐 `32` 倍数、动态连线并裁剪未用可选参考节点；支持 `COLAB_API_URLS` **多台云端实例并行抢单 + 节点掉线自动故障转移** + 断点续跑。

---

## 专项文档导航

* **MiniMax H3 音画一体化实战指南**（模型清单、L4 实测耗时基准、结构化剧本模板、多实例并行与超分）：[docs/MINIMAX_H3_GUIDE.md](docs/MINIMAX_H3_GUIDE.md)
* **从小说到成片生产手册**（人机导演分工、角色三视图定妆、分镜脚本与审片返工）：[docs/PRODUCTION_PLAYBOOK.md](docs/PRODUCTION_PLAYBOOK.md)
* **API 工作流与节点绑定说明**（7 套已验证正式 API JSON 索引）：[workflows/README.md](workflows/README.md)
* **本地配音部署手册（备用 / 捏参考干声用）**（Win11 + RTX 5060 部署 IndexTTS2）：[docs/LOCAL_INDEXTTS2_WIN11.md](docs/LOCAL_INDEXTTS2_WIN11.md)

---

## 快速上手（4 步跑通 MiniMax H3 全链路）

### 第 1 步：初始化本地环境

```bash
# 安装本地调度依赖（推荐使用 uv 或 pip）
pip install requests PyYAML python-dotenv websocket-client

# 复制本地环境变量与工作流映射配置
cp .env.example .env
cp configs/workflow_bindings.example.yaml configs/workflow_bindings.yaml
```

### 第 2 步：准备与预处理素材

将角色三视图、首帧定妆照、参考干声（可选，5~10 秒纯干声）或参考运镜视频放入 `workspace/raw_input/`，运行：

```bash
# 预览清洗与重命名计划
python3 scripts/preprocess_media.py --dry-run

# 按 MiniMax H3 规格清洗（视频 -> 24fps H.264 MP4；音频 -> 32kHz WAV；自动去中文/空格）
python3 scripts/preprocess_media.py --loudnorm
```

清洗后的文件输出至 `workspace/clean_input/`，并生成 `workspace/clean_input/rename_manifest.json`。

### 第 3 步：启动 1~3 台云端 Colab ComfyUI

1. 在 Google Colab 打开 [colab/ShortDrama_ComfyUI.ipynb](colab/ShortDrama_ComfyUI.ipynb)（选择 **L4 / A100 + 高 RAM**；如需加速批量渲染，可在多个标签页同时开 **2~3 台 L4**）。
2. 确认 Colab 左侧 **Secrets** 已添加 `HF_TOKEN` 并开启 Notebook 访问权限。
3. Cell 3 按需勾选模型组（默认已选 `MINIMAX_H3_CORE` + `MINIMAX_H3_FL2VA` + `MINIMAX_H3_REF2VA` + `UPSCALE_2X`；如仅拍首帧 `h3_i2va` 可只留 `CORE` + `FL2VA` 加快启动；如需 8 步快速试戏则勾选 `MINIMAX_H3_TURBO`）。
4. 点「全部运行」，将各台 Cell 4 输出的 `https://xxxx.trycloudflare.com` 填入本地 `.env`：
   - 单实例填 `COLAB_API_URL=https://xxx.trycloudflare.com`
   - 多实例并行用英文逗号拼接填 `COLAB_API_URLS=https://w1.trycloudflare.com,https://w2.trycloudflare.com`

### 第 4 步：云端体检 & 批量渲染

```bash
# 1. 云端连通性、MiniMax H3 节点与模型体检（多实例时自动并发体检并汇总显存）
python3 scripts/check_cloud.py

# 2. 离线演练（校验分镜表、17k+5 帧数换算、32 倍数宽高与节点裁剪，不发网络请求）
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json --dry-run

# 3. 正式提交批量渲染（多实例自动并发抢单，成片拉回 workspace/output/<shot_id>/）
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json

# 4. 仅返工指定镜头 / 重试失败镜头
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json --only-shots H002 --force
python3 scripts/batch_runner.py --storyboard examples/storyboard_h3_sample.json --retry-failed
```

---

## 远程排障常用命令

```bash
# 输出完整 JSON 诊断报告
python3 scripts/check_cloud.py --json

# 远程中断卡死的当前任务 / 清空等待队列（支持多实例同时生效）
python3 scripts/check_cloud.py --interrupt --clear-queue
```

云端日志会按独立 `worker_id` 同步落盘在 Google Drive `MyDrive/ComfyUI_Work/logs/` 下：
- `env_report.json`（GPU、磁盘、PyTorch 与插件版本快照）
- `models_report.json`（模型下载明细）
- `comfyui_<worker_id>.log`（单实例 ComfyUI 完整运行日志）
- `unassign_<worker_id>.log`（空闲看门狗自动关机日志）
- `tunnel_<worker_id>.txt` 与 `active_tunnels.txt`（穿透地址记录）
