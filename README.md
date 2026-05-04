# ComfyUI-Aila-XPU

> 基于 Aila 推理引擎的 ComfyUI 插件，在 Intel Arc 显卡上实现高效的提示词反推（Captioning）。

## 关于 [Aila 引擎 ](https://github.com/Blackwood416/Aila)

Aila 是由 [Blackwood416](https://github.com/Blackwood416) 开发的 Intel Arc 推理引擎，专为 Intel GPU（含 A770、B580 等）优化。
它利用 Intel oneDNN、SYCL 和 Level Zero 技术，在 Arc 显卡上实现了比 llama.cpp 等通用方案更优的推理性能。

感谢 Aila 作者 Blackwood416 在本插件开发过程中的快速响应！

## 简介

本插件通过 [Aila](https://github.com/Blackwood416/Aila) 推理引擎调用 **Qwen3.5 多模态大模型**，在 **Intel Arc 系列显卡（含 B580）** 上对图片进行提示词反推。支持 SD 提示词、中文描述、Danbooru 标签三种输出模式。

## 效果

| 模型 | 显存占用 | 预填充速度 | 解码速度 |
|:----|:--------:|:----------:|:--------:|
| Qwen3.5-4B NF4 | ~4.5 GB | ~1050 tok/s | ~57 tok/s |
| Qwen3.5-0.8B NF4 | ~1.8 GB | ~3870 tok/s | ~144 tok/s |

*以上数据基于 Intel Arc B580 12GB 实测*

## 插件安装

### 方式一：启动器安装（推荐）

打开 ComfyUI 启动器 → 插件管理 → 自定义节点 → 通过 Git URL 安装，输入：

```
https://github.com/allanmeng/ComfyUI-Aila-XPU
```
![演示图](./images/启动器安装插件.png)

安装后还需下载运行时 DLL（首次安装需要）：
1. 打开 [Release 页面](https://github.com/allanmeng/ComfyUI-Aila-XPU/releases) 下载 `aila_runtime_dlls.zip`
2. 解压到 `ComfyUI/custom_nodes/ComfyUI-Aila-XPU/aila_runtime/`（与 `AilaShared.dll` 同目录）

### 方式二：从源码安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/allanmeng/ComfyUI-Aila-XPU
cd ComfyUI-Aila-XPU
pip install -r requirements.txt
```

安装后同样需要下载运行时 DLL（同上，下载 `aila_runtime_dlls.zip` 解压到 `aila_runtime/`）。

### 方式三：网盘下载

[https://pan.quark.cn/s/c793f4fbb990](https://pan.quark.cn/s/c793f4fbb990)

- ComfyUI-Aila-XPU-插件本体.zip
  把里面的`ComfyUI-Aila-XPU 文件夹`整体放到 \ComfyUI\custom_nodes\ 里面

- aila_models.zip 基础模型包
  把里面的`aila 文件夹`整体放到 \ComfyUI\models\ 里面
  这里面是转化好的 `qwen3.5-4b-bnb-nf4-offline` 和 `qwen3.5-0.8b-bnb-nf4-offline`
  如果需要更多的模型，看下面的模型获取章节

- demo_Aila.json  这是个测试工作流文件
  把这个文件放到 \ComfyUI\user\default\workflows\ 里面

### Intel Arc 用户注意

插件依赖 `bitsandbytes`，ComfyUI 启动时会自动安装。
默认安装的 `bitsandbytes` 是 NVIDIA CUDA 版，Intel Arc 用户需手动重装为 XPU 版：

```bash
pip install --force-reinstall bitsandbytes --extra-index-url https://pytorch.org/whl/xpu
```

## 模型获取

模型文件放到 `ComfyUI/models/aila/` 目录下即可使用。

**推荐：直接下载已导出的 NF4 模型（即下即用，无需导出）：**

[https://huggingface.co/collections/Blackwood416/ailas-model-collections](https://huggingface.co/collections/Blackwood416/ailas-model-collections)

[https://pan.quark.cn/s/5d795bb3c417](https://pan.quark.cn/s/5d795bb3c417)

下载后解压到 `ComfyUI/models/aila/` 即可使用。

**或使用导出工具自行导出：**

```bash
python export_model.py --from-hf Qwen/Qwen3.5-4B
```

**网盘中的基础模型文件：** `aila_models.zip` 包含以下已转化好的模型：

- `qwen3.5-4b-bnb-nf4-offline` — 推荐，质量与速度均衡
- `qwen3.5-0.8b-bnb-nf4-offline` — 轻量快速

把里面的 `aila` 文件夹整体放到 `ComfyUI/models/` 里面。

**支持的模型：**

| 模型 | 架构 | 视觉 | 大小 | 推荐场景 |
|:----|:----:|:----:|:----:|:---------|
| Qwen3.5-4B | Hybrid | ✅ 有 | ~3.6 GB | **推荐**，质量与速度均衡 |
| Qwen3.5-0.8B | Hybrid | ✅ 有 | ~942 MB | 轻量快速，质量一般 |
| huihui-Qwen3.5-4B-abliterated | Hybrid | ✅ 有 | ~3.6 GB | Abliterated 版本 |
| Qwen3-4B | Dense | ❌ 无 | ~2.4 GB | 纯文本推理 |
| Qwen3-0.6B | Dense | ❌ 无 | ~525 MB | 纯文本测试 |

## 使用方法

1. **添加 `Aila Model Loader (XPU)` 节点**
   - 选择你下载的模型
   - 点击执行加载

2. **添加 `Aila Engine (XPU)` 节点**
   - 将 Model Loader 的输出连接到 Engine 的 `aila_model` 输入
   - 将需要反推的图片连接到 `images` 输入（可选，不接则为纯文本模式）
   - 配置参数后执行

### 节点参数

| 参数 | 说明 |
|:----|:-----|
| `mode` | 输出模式：`prompt (SD提示词)`、`caption (图片描述)`、`danbooru (Danbooru标签)` |
| `user_prompt` | 自定义用户指令，留空则使用默认 |
| `system_prompt` | 自定义系统提示词，留空则根据 mode 自动选择 |
| `max_tokens` | 最大生成 token 数（默认 256） |
| `temperature` | 采样温度（默认 0.7，越低越稳定） |
| `top_p` / `top_k` | 采样参数 |
| `do_sample` | 启用采样（关闭则为贪心解码） |
| `seed` | 随机种子（0=随机） |
| `memory_cleanup` | `persistent (不释放)` 保留引擎加速连续生成 / `full_cleanup (释放显存, 清理缓存)` 生成后释放 GPU 显存 |

### 三种 mode 说明

| 模式 | 输出语言 | 输出风格 | 适合场景 |
|:----:|:--------:|:---------|:---------|
| **prompt** | 英文 | 一段完整的 SD 提示词 | 直接复制到 positive prompt 使用 |
| **caption** | 中文 | 详细的自然语言描述 | 看图说话、记录内容 |
| **danbooru** | 英文 | 逗号分隔的 Danbooru 风格标签 | 给图片打标签 |

## 技术说明

- 本插件通过 **ctypes** 调用 Aila C API，将图片编码为视觉 token 后让模型生成描述文本
- 图片处理流程：GPU tensor → 临时 PNG 文件 → Aila API 读取 | 生成完成后自动清理
- 支持单张 / 批量图片处理

## 致谢

- [Aila](https://github.com/Blackwood416/Aila) — Intel Arc 推理引擎
- [Qwen](https://github.com/QwenLM/Qwen) — 阿里 Qwen 系列模型
- Intel — XPU PyTorch + oneDNN 加速支持
