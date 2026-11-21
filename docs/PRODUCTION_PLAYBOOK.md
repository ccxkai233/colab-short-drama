# 🎬 从原创小说到 AI 短剧成片实战手册（Production Playbook）

本手册将你与 AI 助手的分工、小说转分镜、角色定妆与音色母带准备、MiniMax H3 音画直出、多实例并行渲染与审片返工串成一条可重复落地的标准流水线。

---

## 一、人机协作分工（导演 vs 技术制片）

* **你（总导演 + 美术总监）**：
  * 守住小说的灵魂与人物气质，提供或拍板**角色三视图定妆照**与关键场景首帧图；
  * 负责审片挑毛病（直接指出哪句情绪不对、哪个微表情太甜/太僵、哪个机位不够近）。
* **AI 助手（技术制片 + 分镜编剧）**：
  * 把小说改写成短剧剧本，并翻译为 MiniMax H3 专用的**结构化时间轴分镜脚本**（含 `[Shot N]`、`(S1)/(S2)` 说话人标签、三层音画分离与精确时间戳）；
  * 归档清洗素材、配置分镜表 JSON 与工作流映射、调度单机或多台云端 Colab 并行渲染；
  * 通过 `--only-shots` 精准返工你点名的单个镜头，或对神级试戏片执行 `upscale_video` 2x 超分。

---

## 二、原创小说 → 视觉圣经 → H3 分镜表（3 步法）

### 第 1 步：建立「角色与场景视觉圣经（Bible）」
开工前，为每位主要角色准备 **1 张高清三视图（正/侧/背）** 及（可选）**5~10 秒纯干声母带**，存入 `workspace/clean_input/`：

| 角色名 | 说话人标签 | 角色三视图（供 R2V `<Picture N>`） | 声音母带（可选，供 R2V `<Audio N>`） |
| :--- | :--- | :--- | :--- |
| 沈清辞（女主） | `(S1)` | `char_shenqingci_3view.png` | `voice_shenqingci_ref.wav` |
| 萧景珩（男主） | `(S2)` | `char_xiaojingheng_3view.png` | `voice_xiaojingheng_ref.wav` |

> 💡 **画质铁律**：有了 MiniMax H3 之后，**绝大多数角色已不再需要专门训练 LoRA**——用高清三视图 + 精准首帧图即可在 `h3_i2va` / `h3_r2v` 中保持极高人脸与服饰一致性。

### 第 2 步：小说章节 → 短剧分集剧本
* **单集规格**：**60 ~ 90 秒**，约 **10 ~ 18 个镜头**，总台词 **180 ~ 280 字**。
* **单镜黄金时长**：每镜控制在 **5 秒（`124` 帧）~ 8 秒（`192` 帧）** 为最佳性价比区间（最长建议不超过 10 秒 `243` 帧），刚好容纳 1~2 句完整对白与自然气口。
* **景别与清晰度协同**：
  * **关系/交代镜头**：双人中景或空镜（`h3_i2va` 锁首帧，3~5 秒即可）；
  * **核心对白/情绪爆发镜**：尽量拆为**单人近景/特写正反打**——脸部占画面面积大 3~4 倍，配合 `768×1344`（竖屏）或 `1344×768`（横屏）+ 原精度 `fp16` Video VAE，清晰度与演技细腻度直逼真剧。

### 第 3 步：按镜头类型选用最佳 MiniMax H3 模式
1. **首帧定妆已知、追求最锐画质与最稳口型** → **`h3_i2va`**（`fl2va` 主权重，单镜或缓慢推拉镜头首选）
2. **需要单段内自动切多机位 / 换新场景 / 挂多角色三视图或锁参考音色** → **`h3_r2v`**（`ref2va` 全参考，支持最多 3 图 + 1 视频 + 2 音频）
3. **快速试戏验证台词断句与微表情** → **`h3_i2va_turbo` / `h3_r2v_turbo`**（8 步 / 4 步 Turbo LoRA，L4 耗时降至 ~1/3）
4. **低清试戏出神级演技不想重抽** → **`upscale_video`**（`RealESRGAN_x2plus` 2x 超分，原表演与原音轨 100% 保留）

> 详细结构化 Prompt 模板、帧数对齐规则与 L4 实测耗时表见 [MINIMAX_H3_GUIDE.md](MINIMAX_H3_GUIDE.md)。

---

## 三、角色参考干声获取与母带规范（可选，用于 R2V 锁声线）

若希望主角在全剧所有镜头里保持绝对统一的专属声线，可为每位主角准备 1 段参考干声，通过 `h3_r2v` 的 `ref_audio_1` 喂给 MiniMax H3（如暂不准备，也可直接让 H3 按声线文字描述原生发声）：

### 4 种获取途径（按商用安全性排序）
1. **【商用首选】AI 语音设计（Voice Design）捏 8~10 秒专属母带**：在 MiniMax 语音 / ElevenLabs / 阿里云百炼 CosyVoice 用人设描述词生成 1 段平静独白 `.wav`，作为永久母带；也可配合本地 **Win11 + RTX 5060 IndexTTS2** 生成任意情绪干声（部署详见 [LOCAL_INDEXTTS2_WIN11.md](LOCAL_INDEXTTS2_WIN11.md)）。
2. **免费开源纯净语料**：ModelScope / Hugging Face 开源纯净中文数据集（如 `AISHELL-1/3`、`MagicData`）。
3. **自录 / 朋友录 10 秒**：安静房间贴近吸音物平稳朗读，轻度降噪，版权完全归自己。
4. **【仅限个人练手，商用严禁】影视剥离干声**：UVR5（BS-RoFormer + De-Reverb）去混响去 BGM；正式商用切勿克隆未经授权的真人声音。

### 参考母带黄金规格
* **时长**：**5 ~ 10 秒**最佳；单人纯干声、无 BGM、无回声、无爆麦；
* **情绪基调**：务必选**自然、平静、吐字清晰**的语调，具体情绪交给 H3 提示词或 IndexTTS2 情感控制去演绎；
* **预处理**：放入 `workspace/raw_input/` 后跑 `python3 scripts/preprocess_media.py --loudnorm`，自动洗成 **32kHz PCM_16 WAV**。

---

## 四、批量加速生产与审片精准返工

### 1. 多台 Colab L4 并行跑整集（耗时除以 N，总算力不变）
在浏览器同时开 2~3 个 Colab 标签页运行 `ShortDrama_ComfyUI.ipynb`，将所有地址用逗号填进 `.env` 的 `COLAB_API_URLS`：

```bash
# 一键并发体检所有云端实例
python3 scripts/check_cloud.py

# 多实例动态抢单渲染（某台断线自动由存活节点接管）
python3 scripts/batch_runner.py --storyboard workspace/storyboards/ep01_scene01_h3.json
```

### 2. 单镜头审片返工命令速查
审片发现个别镜头不满意（例如微表情偏甜、语气偏平、需要改 `seed` 或台词节奏），只改分镜表里对应几行，精准返工不影响已通过镜头：

```bash
# 仅重跑指定镜头
python3 scripts/batch_runner.py \
  --storyboard workspace/storyboards/ep01_scene01_h3.json \
  --only-shots EP01_S01_SHEN_916 \
  --force

# 仅重试上一轮报错失败的镜头
python3 scripts/batch_runner.py \
  --storyboard workspace/storyboards/ep01_scene01_h3.json \
  --retry-failed
```

---

## 五、备用路线：传统分步管线（FLUX LoRA + Wan 2.2 + LatentSync）

当个别特殊镜头需要极致的传统可控性（例如必须使用自定义画风 FLUX 角色 LoRA，或必须严格套用本地 IndexTTS2 精确到毫秒的外部成品干声对口型）时，可随时切回仓库内置的备用链路：
- **生图**：`task_type: t2i_flux`（[workflows/t2i_flux_api.json](../workflows/t2i_flux_api.json)）；如需自训 `FLUX.1-dev` 角色 LoRA，可用 `ostris/ai-toolkit`（`linear=16`、`lr=1e-4`、`steps=1250~1500`）训练 15~25 张“五官不变、角度/服装/光线/背景全变”的训练集并存入 `MyDrive/ComfyUI_Work/loras/`。
- **无声视频 + 外部配音对口型**：`task_type: i2v_wan22` → `preprocess_media.py --profile latentsync`（洗成 25fps + 16kHz 单声道 WAV）→ `task_type: lipsync`。
