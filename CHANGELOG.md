# Changelog

## v0.1.7.2 (2026-07-25)

### 引擎升级
- **引擎升级到 Aila v0.1.7**：进程隔离架构（AilaShared.dll 为轻量代理，AilaWorker.exe 独立进程推理）
- **目录结构调整**：AilaShared.dll 从 `aila_runtime/` 移到插件根目录
- 新增 `AILA_RUNTIME_DLL_DIR` 环境变量支持
- 移除 `os.add_dll_directory("aila_runtime")` 调用（避免破坏进程隔离）
- 配置 `dll_path` 默认路径更新
- 新增 `git` 追踪：`AilaShared.dll`（根目录）+ `aila_runtime/AilaWorker.exe`

### 文档
- README / README_EN 更新：v0.1.7 新目录结构、环境变量说明、安装方式更新

## v0.1.7.1 (2026-07-25)

### 修复
- 移除启动时自动 pip install bitsandbytes 的逻辑
- bitsandbytes 检测改为仅提示状态，不影响 Aila 引擎推理
- 修复自动安装可能导致用户 PyTorch 环境被污染的问题
- 移除 `subprocess`、`sys`、`os` 三个不再需要的 import
- 适配新版 bitsandbytes（`COMPILED_WITH_CUDA` 属性已移除，改用 `BNB_BACKEND`）

## v0.1.7 (2026-06-06)

### 插件
- **引擎升级到 Aila v0.1.6**：替换 Aila.exe + AilaShared.dll
- **TTS Synthesizer 增强**：
  - 新增 `speaker_name` 参数（CustomVoice 预设音色：Ryan/Vivian 等 9 种）
  - 新增 `ref_audio` 输入（Base 模型语音克隆）
  - 新增 `instruct` 参数（VoiceDesign 风格指令）
  - `model_type` 自动识别，参数互斥生效
  - 预设音色/风格指令支持自动分段合成
- 新增 C API 绑定：`aila_extract_speaker_embedding` + `aila_synthesize`
- **ASR Transcriber 增强**：
  - 新增 `force_aligner_model` 参数（Qwen3-ForceAligner 强制对齐）
  - 新增 `SUBTITLES` 输出（SRT 格式字幕）
  - 新增 `subtitle_mode` 参数（按断句/按词/按字三种粒度）
  - 长音频自动分块对齐（2 分钟/块，支持 5 分钟以上音频）
- 修复 `io.NodeOutput` 多输出返回值错误
- 修复 ForceAligner 逐字对齐时标点缺失导致的断句检测失败
- 新增依赖：`jieba`（中文分词，用于字幕生成）
- 调整 ASR 默认参数：`max_tokens=1024`（勿超，ASR 模型 max_seq=2048）
- README 更新：新增 TTS/ASR 高级功能说明
- 修复首次安装时 `models/aila/asr/` 目录无法自动创建的路径错误
- 新增 `models/aila/llm/`、`models/aila/tts/` 目录自动创建（`mkdir(parents=True)`）

### 引擎
- Aila v0.1.5 / v0.1.6
- 新增 CustomVoice 模型支持（预设音色）
- 新增 VoiceDesign 模型支持（风格指令）
- 新增 TTS 流式合成 API
- 新增 ForceAligner 支持
- 修复 speaker embedding 提取问题
- TTS 性能优化

---

## v0.1.6 (2026-06-02)

### 插件
- **新增 ASR 语音转录**：Aila ASR Loader + Aila ASR Transcriber 两个节点
- **新增 TTS 语音合成**：Aila TTS Loader + Aila TTS Synthesizer 两个节点
- 节点命名统一，执行节点带模型类型前缀（LLM / ASR / TTS）
- 修复 `aila_transcribe` ctypes 指针处理错误（0xc0000374 堆损坏崩溃）
- 修复 `aila_free_string(lang_ptr.value)` 访问违规崩溃
- ASR Transcriber 移除 `asr_past` 参数，`asr_segment` 默认改为 -1（-1/0=不分段）
- TTS Synthesizer 支持 `auto_segment` 按句分段合成 + `max_new_tokens` 控制
- TTS 模型扫描 `models/aila/tts/`，支持 Qwen3-TTS-12Hz-0.6B/1.7B-Base
- ASR 模型扫描 `models/aila/asr/`，支持 Qwen3-ASR-0.6B/1.7B（NF4 / BF16）
- `.gitignore` 增加 `Aila.exe` 追踪
- README 大更新：新增 ASR/TTS 内容、模型表格带下载链接、目录说明

### 引擎
- Aila v0.1.4
- 新增 Qwen3-TTS Base 模型推理支持（CLI + API）
- 新增 Qwen3-ASR NF4 量化支持
- 新增音频转录流式支持
- ASR 转录性能 10x 提升（1.7B 模型超实时）
- 集成 llama.cpp jinja 引擎，支持 OpenAI 兼容 JSON 格式输入

---

### 插件
- `requirements.txt` 首选 XPU 索引，新用户一步到位安装 XPU 版 bitsandbytes
- `__init__.py` 新增 Intel Arc 自动检测 + 自动修复（为已有 CUDA 版 bnb 的用户兜底）
- 新手安装零门槛，无需手动命令

### 引擎
- Aila v0.1.2（未更新）

---

## v0.1.5 (2026-05-24)

### 插件
- `requirements.txt` 首选 XPU 索引，新用户一步到位安装 XPU 版 bitsandbytes
- `__init__.py` 新增 Intel Arc 自动检测 + 自动修复（为已有 CUDA 版 bnb 的用户兜底）
- 新手安装零门槛，无需手动命令

### 引擎
- Aila v0.1.2（未更新）

---

## v0.1.4 (2026-05-12)

### 插件
- 更新 AilaShared.dll 至 v0.1.2

### 引擎 (Aila v0.1.2)
- 解码速度提升超过 10%（通过手动展开反量化算子 + vec8 权重打包）
- A770 16G 最佳性能：prefill 1649 tok/s, decode 58 tok/s
- 新增全局日志级别控制
- 修复 CLI `--messages-json` 输出中 `<think>` 标签消失的问题
- 优化启动预热性能

---

## v0.1.3 (2026-05-07)

### 插件
- 新增 `rep_penalty` / `pres_penalty` / `freq_penalty` 三个惩罚参数（float 输入）
- 新增 `enable_thinking` 开关（关闭后自动追加 `/no_think` 抑制思维链）
- 新增 `debug` 开关（开启后输出 Token ID 调试日志）
- Captioner 节点输入参数已重排，惩罚参数在 seed 下方

### 引擎
- Aila v0.1.1（未更新）

---

## v0.1.2 (2026-05-04)

### 插件
- 节点名改为 `Aila Model Loader (XPU)` / `Aila Engine (XPU)`
- `memory_cleanup` 改为下拉选项：`persistent (不释放)` / `full_cleanup (释放显存, 清理缓存)`
- 引擎销毁后自动重建，不再报"模型未加载"
- 纯文本模式支持（不接图片即可使用）
- mode 下拉菜单增加中文说明
- AilaShared.dll 纳入 git 追踪，其他运行时 DLL 通过 Release 分发

### 引擎
- Aila v0.1.1
- 修复输出中的思维链（CoT）问题
- 新增 `/think` 后缀命令
- 修复 B580 上 subgroup size 兼容性问题

---

## v0.1.1 (2026-05-02)

### 插件
- `memory_cleanup` 功能（布尔开关）
- IMAGE 输入改为可选，支持纯文本模式
- 添加 Intel Arc 用户安装说明

### 引擎
- Aila v0.1.0
- Qwen3.5-4B / 0.8B NF4 量化推理
- bitsandbytes XPU 后端

---

## v0.1.0 (2026-05-02)

### 插件
- 首个公开发布版本
- 支持 Aila Model Loader + Aila Captioner 两个节点
- 三种 mode：prompt / caption / danbooru
- 支持单张/批量图片处理
- 支持 Intel Arc B580 (12GB)

### 引擎
- Aila v0.1.0
