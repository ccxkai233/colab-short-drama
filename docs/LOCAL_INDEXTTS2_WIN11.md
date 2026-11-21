# 🎙️ 本地语音克隆与配音部署手册（IndexTTS2 · Windows 11 + RTX 5060 8GB）

## 为什么把配音放本地，而不是放 Colab？

1. **零云端算力费**：短剧配音需要高频试听、改标点、调情绪与停顿，放本地 RTX 5060（8GB）运行零计算单元消耗，1~3 秒出一条台词。
2. **避免依赖冲突**：TTS 模型依赖与云端 ComfyUI / LatentSync 独立隔离，互不干扰。
3. **贴合工业化短剧节奏**：先在本地敲定全部干声台词（决定每个镜头的精确秒数）→ `preprocess_media.py` 洗成 16kHz 标准 `.wav` → 再开 Colab 批量渲染视频与对口型，跑完自动释放算力。

---

## 一、前置准备（Windows 11）

- [ ] **NVIDIA 显卡驱动 ≥ 570**（RTX 5060 为 Blackwell 架构 `sm_120`，需驱动支持 CUDA 12.8）
- [ ] 安装 **Git**（安装向导勾选 **Git LFS**）
- [ ] 安装 **uv**（在 PowerShell 执行，装完重开一个 PowerShell 窗口）：
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

---

## 二、5 步部署 IndexTTS2

### 第 1 步：克隆仓库（务必放在纯英文、无空格路径）
```powershell
mkdir D:\AI -Force
cd D:\AI

git lfs install
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull
```

### 第 2 步：安装依赖（Windows 避坑）
> ⚠️ **避坑提示**：Windows 原生编译 DeepSpeed 容易报错，请使用 `--extra webui`，不要使用 `--all-extras`；且**不要手动激活 `.venv`**，后续全程用 `uv run` 启动即可。

```powershell
uv sync --extra webui
```

### 第 3 步：RTX 5060（`sm_120` / CUDA 12.8）兼容性校验
先运行官方 GPU 检测：
```powershell
uv run tools/gpu_check.py
```

- 若正常识别到 `RTX 5060` 且无 `sm_120` / `no kernel image is available for execution on the device` 报错，直接跳到第 4 步。
- 若报架构不兼容或 PyTorch 版本过老，在当前目录执行升级至 **CUDA 12.8 (`cu128`)**：
  ```powershell
  uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```

### 第 4 步：下载 IndexTTS-2 模型权重（约 6GB）
```powershell
# 国内网络若直连 Hugging Face 较慢，可先取消下面这行注释启用镜像：
# $env:HF_ENDPOINT="https://hf-mirror.com"

uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```
*(备用：也可通过 ModelScope 下载 `uv run modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints`)*

### 第 5 步：启动本地 GUI 面板（8GB 显存必加 `--fp16`）
```powershell
uv run webui.py --fp16
```

- 首次启动会自动补拉少量辅助特征编码器。
- 启动完成后，在浏览器打开 **`http://127.0.0.1:7860`** 即可进入 Gradio 可视化操作面板：
  1. 上传 5~10 秒角色干声参考音频（定音色）；
  2. 输入台词文本；
  3. 调节情绪方式（情绪参考音频 / 情绪滑块 / 情绪描述词）；
  4. 点击生成、在线试听、满意后下载 `.wav`。

> 💡 **`--fp16` 说明**：在 8GB RTX 5060 上可将显存占用压在 **4~5GB** 且推理速度翻倍，听感几乎无损。

---

## 三、创建双击一键启动脚本（可选推荐）

在 `D:\AI\index-tts\` 根目录新建 `start_indextts2.bat`，以后直接双击即可启动面板：

```bat
@echo off
cd /d %~dp0
uv run webui.py --fp16
pause
```

---

## 四、与 Short-Drama-Factory 管线的衔接

1. **手工试音阶段**：把在 `http://127.0.0.1:7860` 挑好的台词 `.wav` 放入本项目的 `workspace/raw_input/`，然后运行本地预处理：
   ```bash
   python3 scripts/preprocess_media.py --fps 25 --loudnorm
   ```
   脚本会自动将录音统一清洗为 LatentSync-1.6 标准的 **16kHz 单声道 16-bit PCM `.wav`**，并输出文件名映射表 `workspace/clean_input/rename_manifest.json`。
2. **后续批量自动化扩展**：由于 `7860` 端口的 Gradio WebUI 同时自带本地 HTTP API，待面板跑通后可随时按需追加 `configs/voices.yaml` + `scripts/generate_tts.py`，实现「剧本/分镜表 → 自动调本地 7860 批量出干声 → 自动 16kHz 洗帧 → 云端批量对口型」全链路自动化。
