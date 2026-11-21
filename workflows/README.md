# 工作流目录使用说明（API Format）

本目录用于存放从 ComfyUI 导出的 **API 格式工作流 JSON**（供 `scripts/batch_runner.py` 批量替换参数并提交云端渲染）。

## 如何正确导出 API 格式工作流？

1. 打开云端 ComfyUI 网页（`https://xxxx.trycloudflare.com`），先把你的工作流手工跑通 1 次。
2. 点击右上角齿轮（Settings）→ 开启 **Enable Dev mode Options**。
3. 在菜单中点击 **Save (API Format)**（注意：不要只点普通 `Save`，普通保存是 UI 布局格式，无法直接用于 `/prompt` 接口）。
4. 将导出的 `.json` 保存到本目录（例如 `workflows/my_lipsync_api.json`）。
5. 复制 `configs/workflow_bindings.example.yaml` 为 `configs/workflow_bindings.yaml`，对照你导出的 JSON 修改里面的 `node_id` 和 `input` 字段名即可。

## 预置示例模板说明

- `t2i_flux_api.example.json`：FLUX.1-dev FP8 文生图 + LoRA 节点结构模板
- `i2v_wan22_api.example.json`：Wan 2.2 14B MoE（高噪/低噪双阶段采样）图生视频结构模板
- `lipsync_latentsync_api.example.json`：LatentSync-1.6 对口型结构模板

> 提示：不同版本的插件节点类名或输入键名可能略有差异（例如 `VHS_LoadVideo` / `LoadVideo`、`LatentSyncNode` / `LatentSyncEnhancedNode`）。以你云端界面实际跑通后导出的 API JSON 为准，配合 `workflow_bindings.yaml` 解耦映射即可。
