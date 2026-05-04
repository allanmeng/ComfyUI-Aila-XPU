# ComfyUI-Aila-XPU

> ComfyUI captioning plugin powered by Aila inference engine, optimized for Intel Arc GPUs.

[**中文文档**](./README.md)

## About [Aila](https://github.com/Blackwood416/Aila)

Aila is an Intel Arc inference engine developed by [Blackwood416](https://github.com/Blackwood416), optimized for Intel GPUs including A770 and B580. It leverages Intel oneDNN, SYCL, and Level Zero technologies to deliver superior inference performance on Arc GPUs.

Special thanks to Blackwood416 for the rapid responses during plugin development!

## Overview

This plugin uses the Aila C API to run **Qwen3.5 multimodal models** on **Intel Arc GPUs** for image captioning. It supports three output modes: SD prompts, Chinese descriptions, and Danbooru tags.

## Performance

| Model | VRAM Usage | Prefill Speed | Decode Speed |
|:----|:--------:|:----------:|:--------:|
| Qwen3.5-4B NF4 | ~4.5 GB | ~1050 tok/s | ~57 tok/s |
| Qwen3.5-0.8B NF4 | ~1.8 GB | ~3870 tok/s | ~144 tok/s |

*Benchmarked on Intel Arc B580 12GB*

## Installation

### Method 1: ComfyUI Manager (Recommended)

Open ComfyUI Manager → Install Custom Nodes → Install via Git URL, enter:

```
https://github.com/allanmeng/ComfyUI-Aila-XPU
```

After installation, download the runtime DLLs (first-time setup):
1. Go to [Release page](https://github.com/allanmeng/ComfyUI-Aila-XPU/releases) and download `aila_runtime_dlls.zip`
2. Extract to `ComfyUI/custom_nodes/ComfyUI-Aila-XPU/aila_runtime/` (same directory as `AilaShared.dll`)

### Method 2: Git Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/allanmeng/ComfyUI-Aila-XPU
cd ComfyUI-Aila-XPU
pip install -r requirements.txt
```

Then download runtime DLLs as described above.

### Method 3: Direct Download

[https://pan.quark.cn/s/c793f4fbb990](https://pan.quark.cn/s/c793f4fbb990) (Cloud drive, Chinese)

- `ComfyUI-Aila-XPU-插件本体.zip` — Plugin package with runtime DLLs included
  Extract `ComfyUI-Aila-XPU` folder to `\ComfyUI\custom_nodes\`

- `aila_models.zip` — Pre-exported models
  Extract the `aila` folder to `\ComfyUI\models\`
  Includes `qwen3.5-4b-bnb-nf4-offline` and `qwen3.5-0.8b-bnb-nf4-offline`

- `demo_Aila.json` — Test workflow
  Place in `\ComfyUI\user\default\workflows\`

### Intel Arc Users

This plugin depends on `bitsandbytes`. ComfyUI auto-installs dependencies on startup, but the default installation is the CUDA version. Intel Arc users need to manually reinstall the XPU version:

```bash
pip install --force-reinstall bitsandbytes --extra-index-url https://pytorch.org/whl/xpu
```

## Getting Models

Place model files in `ComfyUI/models/aila/`.

**Recommended: Download pre-exported NF4 models (ready to use):**

[https://huggingface.co/collections/Blackwood416/ailas-model-collections](https://huggingface.co/collections/Blackwood416/ailas-model-collections)

Extract to `ComfyUI/models/aila/` and you're ready to go.

**Or export manually:**

```bash
python export_model.py --from-hf Qwen/Qwen3.5-4B
```

**Supported Models:**

| Model | Architecture | Vision | Size | Use Case |
|:----|:----:|:----:|:----:|:---------|
| Qwen3.5-4B | Hybrid | ✅ Yes | ~3.6 GB | **Recommended**, quality & speed |
| Qwen3.5-0.8B | Hybrid | ✅ Yes | ~942 MB | Lightweight |
| huihui-Qwen3.5-4B-abliterated | Hybrid | ✅ Yes | ~3.6 GB | Abliterated version |
| Qwen3-4B | Dense | ❌ No | ~2.4 GB | Text-only |
| Qwen3-0.6B | Dense | ❌ No | ~525 MB | Text-only testing |

## Usage

1. **Add `Aila Model Loader (XPU)` node**
   - Select your downloaded model
   - Execute to load

2. **Add `Aila Engine (XPU)` node**
   - Connect the Model Loader output to the Engine's `aila_model` input
   - Connect your image to `images` input (optional; leave empty for text-only mode)
   - Configure parameters and execute

### Node Parameters

| Parameter | Description |
|:----|:-----|
| `mode` | Output mode: `prompt (SD prompt)` / `caption (description)` / `danbooru (tags)` |
| `user_prompt` | Custom user instruction |
| `system_prompt` | Custom system prompt |
| `max_tokens` | Max tokens to generate (default 256) |
| `temperature` | Sampling temperature (default 0.7) |
| `top_p` / `top_k` | Sampling parameters |
| `do_sample` | Enable sampling (disable for greedy decoding) |
| `seed` | Random seed (0=random) |
| `memory_cleanup` | `persistent` keeps engine loaded / `full_cleanup` frees GPU memory after generation |

### Output Modes

| Mode | Language | Style | Use Case |
|:----:|:--------:|:------|:---------|
| **prompt** | English | Full SD prompt | Copy directly to positive prompt |
| **caption** | Chinese | Detailed natural language description | Image description |
| **danbooru** | English | Comma-separated Danbooru-style tags | Image tagging |

## Technical Notes

- Uses **ctypes** to call the Aila C API, encoding images as vision tokens for the model
- Image pipeline: GPU tensor → temporary PNG file → Aila API | Auto-cleaned after generation
- Supports single and batch image processing

## Credits

- [Aila](https://github.com/Blackwood416/Aila) — Intel Arc inference engine
- [Qwen](https://github.com/QwenLM/Qwen) — Alibaba Qwen models
- Intel — XPU PyTorch + oneDNN acceleration
