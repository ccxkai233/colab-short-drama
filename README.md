# 🎬 Short-Drama-Factory（本地大脑 + 云端肌肉 AI 短剧自动化生产管线）

一套面向 **本地 CLI 调度 + Google Colab Pro+（L4 / A100 / G4）ComfyUI 渲染** 的轻量化、高可维护 AI 短剧自动化生产管线。

---

## 核心架构特点

0. **本地硬件 vs 云端算力最优分工**
   - **本地主机（如 Win11 + RTX 5060 8GB + 32GB DDR5）**：负责 **IndexTTS2 音色克隆与台词配音**（高频试听、零云端扣费，详见 [docs/LOCAL_INDEXTTS2_WIN11.md](docs/LOCAL_INDEXTTS2_WIN11.md)）、素材批量预处理洗帧、API 批量调度与后期剪辑。
   - **云端 Colab Pro+（L4 / A100 / G4）**：专心负责高显存批量渲染（FLUX 分镜底图、Wan 2.2 14B FP8 图生视频、LatentSync-1.6 对口型），队列一空自动释放算力。

1. **动静分离存储**
   - **Google Drive（`MyDrive/ComfyUI_Work/`）**：仅持久化 `input/`、`output/`、`workflows/`、`loras/`、`scripts/`、`logs/`。
   - **Colab 高速本地盘（`/content/ComfyUI`）**：存放 ComfyUI 本体与几十 GB 基础大模型，开机用 `HF_HUB_ENABLE_HF_TRANSFER=1` 满速直拉，关机自动清空。
2. **云端已知坑位全自动处理**
   - 使用 `uv` 极速安装依赖，同时过滤 `torch/torchvision/torchaudio` 防止破坏 Colab 预装 CUDA 环境。
   - 显式补全 `guardskill/comfyui-latentsync-enhanced` 缺失依赖（`insightface`、`onnxruntime-gpu`、`librosa`、`DeepCache`、`imageio`、`imageio-ffmpeg`、`soundfile`、`matplotlib`、`more-itertools`、`ffmpeg-python`）。
   - 自动将 `stabilityai/sd-vae-ft-mse` 拉取至 `models/checkpoints/LatentSync-1.6/vae/` 满足离线加载。
   - 内置 `colab/patch_latentsync.py` 运行态探测补丁：若当前云端 PyTorch/torchaudio/torchvision 触发 `torchaudio.save` 或 `torchvision.io.read_video` 报错，自动备份并替换为 `soundfile` + `cv2`。
   - 内置 `colab/idle_unassign.py` 空闲看门狗：任务队列清空并空闲满设定时间后自动执行 `google.colab.runtime.unassign()`，防止空转扣计算单元。
3. **本地大脑 CLI 工具链**
   - `scripts/preprocess_media.py`：批量把录音洗成 16kHz 单声道 16-bit PCM `.wav`（自动跳过坏帧），视频帧率对齐（默认 25fps），自动清洗中文/空格文件名并生成映射表。
   - `scripts/check_cloud.py`：一键远程体检云端 GPU、插件加载、模型目录与最近执行报错。
   - `scripts/batch_runner.py`：读取 CSV/JSON 分镜表，自动上传素材、替换 API 工作流参数、排队渲染、断点续跑并自动拉回成片到本地归档。

---

## 相关专项文档

* **从小说到成片实战手册**（导演/AI 分工、小说转分镜、角色干声获取、角色 LoRA 制作与看图调参）：[docs/PRODUCTION_PLAYBOOK.md](docs/PRODUCTION_PLAYBOOK.md)
* **本地配音部署手册**（Win11 + RTX 5060 部署 IndexTTS2）：[docs/LOCAL_INDEXTTS2_WIN11.md](docs/LOCAL_INDEXTTS2_WIN11.md)
* **API 工作流导出与绑定指南**：[workflows/README.md](workflows/README.md)

---

## 快速上手（4 步跑通全链路）

### 第 1 步：初始化本地环境

```bash
# 安装本地调度依赖（推荐使用 uv 或 pip）
uv pip install -r <(python3 -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
# 或直接用 pip：
pip install requests PyYAML python-dotenv websocket-client

# 复制本地环境变量与工作流映射配置
cp .env.example .env
cp configs/workflow_bindings.example.yaml configs/workflow_bindings.yaml
```

### 第 2 步：本地配音 & 预处理素材

1. **生成角色干声台词（本地 RTX 5060）**：按 [docs/LOCAL_INDEXTTS2_WIN11.md](docs/LOCAL_INDEXTTS2_WIN11.md) 启动本地 IndexTTS2 GUI 面板（`http://127.0.0.1:7860`），克隆角色音色、试听调好情绪并导出干声 `.wav`。
2. **批量洗帧与对齐**：将干声录音、驱动视频和分镜底图放进 `workspace/raw_input/`，运行：

```bash
# 预览清洗与重命名计划
python3 scripts/preprocess_media.py --dry-run

# 执行全量清洗（录音 -> 16kHz mono WAV；视频 -> 25fps H.264 MP4；自动去中文/空格）
python3 scripts/preprocess_media.py --fps 25 --loudnorm
```

清洗后的文件会输出至 `workspace/clean_input/`，同时生成 `workspace/clean_input/rename_manifest.json` 方便对照原始文件名。

### 第 3 步：启动云端 Colab ComfyUI

1. 将本项目的 `colab/*.py` 与 `configs/models.yaml` 上传至 Google Drive 的 `MyDrive/ComfyUI_Work/scripts/`（或将整个仓库推到你的 GitHub，在 Notebook Cell 1 填入仓库地址自动同步）。
2. 在 Google Colab 打开 `colab/ShortDrama_ComfyUI.ipynb`（选择 L4 / A100 / G4 + High-RAM）。
3. 确认 Colab 左侧 **Secrets** 已添加 `HF_TOKEN` 并开启访问权限。
4. 依次运行 **Cell 1 → Cell 2 → Cell 3（按需勾选模型组）→ Cell 4**。
5. 复制 Cell 4 打印出的 `https://xxxx.trycloudflare.com` 地址，填入本地 `.env` 的 `COLAB_API_URL`。

### 第 4 步：云端体检 & 批量渲染

```bash
# 1. 云端连通性、插件与模型体检
python3 scripts/check_cloud.py

# 2. 离线演练（检查分镜表与工作流节点映射是否正确，不发网络请求）
python3 scripts/batch_runner.py --storyboard examples/storyboard_sample.csv --dry-run

# 3. 正式批量提交渲染并自动拉回成片到 workspace/output/<shot_id>/
python3 scripts/batch_runner.py --storyboard examples/storyboard_sample.csv

# 4. 仅重跑上次失败的镜头（断点续跑）
python3 scripts/batch_runner.py --storyboard examples/storyboard_sample.csv --retry-failed
```

---

## 远程排障常用命令

```bash
# 输出完整 JSON 诊断报告（可直接发给 AI 助手定位节点/模型/报错堆栈）
python3 scripts/check_cloud.py --json

# 远程中断卡死的当前任务 / 清空等待队列
python3 scripts/check_cloud.py --interrupt --clear-queue
```

云端所有启动日志与环境快照也会同步落盘在 Google Drive 的 `MyDrive/ComfyUI_Work/logs/` 下：
- `env_report.json`（GPU、磁盘、PyTorch 与插件版本快照）
- `patch_report.json`（LatentSync 兼容性探测与补丁结果）
- `models_report.json`（模型下载明细）
- `comfyui.log`（ComfyUI 完整运行日志）
- `tunnel_url.txt`（本次启动生成的公网穿透地址）
