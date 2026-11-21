# 工作流目录使用说明（API Format · ComfyUI 0.46.0+）

本目录存放供 `scripts/batch_runner.py` 批量调用的 **ComfyUI API 格式工作流 JSON**。所有主力工作流均已按云端 **ComfyUI `0.46.0` 原生节点接口**实测校准，无需手动连线即可直接跑通。

---

## 一、已校准正式工作流清单（与 `configs/workflow_bindings.example.yaml` 一一对应）

### 主力：MiniMax H3 音画一体化工作流（零第三方插件）
| 文件 | 对应 `task_type` | 主权重 | 用途与说明 |
| :--- | :--- | :--- | :--- |
| `h3_fl2va_api.json` | `h3_t2va` / `h3_i2va` | `minimax_h3_fl2va_pruned_int8_convrot` + `fp16` Video VAE | 文生音画 / 首帧图生音画 / 首尾帧音画（20~25 步定稿，原生 `BasicGuider` 无负面词） |
| `h3_r2v_api.json` | `h3_r2v` | `minimax_h3_ref2va_pruned_int8_convrot` + `fp16` Video VAE | 多模态全参考（支持动态挂载最多 3 图 + 1 视频 + 可选视频原声 + 2 独立音频） |
| `h3_fl2va_turbo_api.json` | `h3_t2va_turbo` / `h3_i2va_turbo` | FL2VA + `fl2v_turbo_8step` LoRA | 8 步快速试戏草稿（L4 耗时降至约 1/3） |
| `h3_r2v_turbo_api.json` | `h3_r2v_turbo` | REF2VA + `ref2v_turbo_4step` LoRA | 4 步 R2V 快速试戏草稿 |
| `upscale_video_2x_api.json` | `upscale_video` | `RealESRGAN_x2plus.pth`（64 MB） | 2x 原生视频高清超分，自动透传原音轨与原帧率 |

### 配套定妆照生图 & 传统备用链路
| 文件 | 对应 `task_type` | 说明 |
| :--- | :--- | :--- |
| `t2i_flux_api.json` | `t2i_flux` | FLUX.1-dev FP8 + `FluxGuidance` 文生图（已云端实测通过） |
| `i2v_wan22_api.json` | `i2v_wan22` | Wan 2.2 14B FP8 双阶段 MoE + `ModelSamplingSD3` + `CreateVideo` + `SaveVideo`（已云端实测通过） |
| `*_api.example.json` | 参考模板 | 早期示例结构模板（保留备查，如含 `LoraLoader` 的 FLUX 挂 LoRA 模板与 LatentSync 模板） |

---

## 二、动态可选参考与自动对齐机制（`workflow_bindings.yaml`）

`scripts/batch_runner.py` 针对 MiniMax H3 扩展了以下绑定能力，可让同一套 JSON 自动适配不同组合的输入：
- `type: "h3_seconds"` / `"h3_frames"`：自动将秒数（24fps）或帧数向上对齐到 MiniMax H3 的 **`17k + 5`** 网格（如 `5s → 124`、`8s → 192`、`10s → 243`）。
- `align: 32`：自动将 `width` / `height` 对齐为 `32` 的倍数。
- `connect_to` + `prune_nodes`：分镜行提供了 `first_frame` / `ref_image_1..3` / `ref_video_1` / `ref_audio_1..2` 时自动上传并连线；未提供时自动把未使用的 `__OPTIONAL_*__` 占位加载节点从提交载荷中裁掉，避免 ComfyUI 校验报错。
- `boolean_connect`：当 `ref_video_1_use_audio: true` 时自动把 `GetVideoComponents` 的音频口连到 `MiniMaxH3ReferenceToVideo.ref_video_audios.ref_video_audio_0`。

---

## 三、如何导出并接入你自己的自定义 API 工作流？

1. 打开云端 ComfyUI 网页（`https://xxxx.trycloudflare.com`），先把自定义工作流手动跑通 1 次。
2. 点击右上角齿轮（Settings）→ 开启 **Enable Dev mode Options**。
3. 在菜单中点击 **Save (API Format)**（不要点普通 `Save`，普通保存是 UI 布局格式，无法直接用于 `/prompt` 接口）。
4. 将导出的 `.json` 保存到本目录，并在 `configs/workflow_bindings.yaml` 中新增或更新对应 `task_type` 的 `node_id` 与 `input` 映射即可。
