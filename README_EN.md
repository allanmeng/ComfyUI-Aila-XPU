# ComfyUI-Aila-XPU

> A ComfyUI plugin powered by the Aila inference engine, enabling efficient LLM captioning, ASR speech-to-text, and TTS text-to-speech on Intel Arc GPUs.
>
> [**中文文档**](./README.md)

## About [Aila Engine](https://github.com/Blackwood416/Aila)

Aila is an Intel Arc inference engine developed by [Blackwood416](https://github.com/Blackwood416), optimized for Intel GPUs (A770, B580, etc.).
Leveraging Intel oneDNN, SYCL and Level Zero, it delivers superior inference performance on Arc GPUs compared to general-purpose solutions like llama.cpp.

Special thanks to Blackwood416 for the quick responses during plugin development!

## Overview

This plugin uses the [Aila](https://github.com/Blackwood416/Aila) inference engine to run **Qwen models** on **Intel Arc GPUs (B580 included)**, providing three features:

- **LLM Captioner** — Image captioning (SD prompts, Chinese descriptions, Danbooru tags), also supports pure text Q&A
- **ASR Transcriber** — Speech-to-text, supports short audio and long audio with segmentation
- **TTS Synthesizer** — Text-to-speech, supports auto-segmentation for long text

## Benchmarks

### LLM Captioner

| Model | VRAM | Prefill Speed | Decode Speed |
|:----|:--------:|:----------:|:--------:|
| Qwen3.5-4B NF4 | ~4.5 GB | ~1050 tok/s | ~57 tok/s |
| Qwen3.5-0.8B NF4 | ~1.8 GB | ~3870 tok/s | ~144 tok/s |

### ASR Transcriber (77s audio, Aila v0.1.4)

| Model | VRAM | Latency | Speed |
|:----|:--------:|:----:|:----:|
| Qwen3-ASR-1.7B BF16 | ~7.3 GB | 19.7s | 3.9x |
| Qwen3-ASR-1.7B BNB NF4 | **~3.4 GB** | **11.4s** | **6.8x** |

### TTS Synthesizer

| Model | VRAM |
|:----|:--------:|
| Qwen3-TTS-12Hz-0.6B-Base | ~2.2 GB |
| Qwen3-TTS-12Hz-1.7B-Base | ~6.7 GB |

*All data measured on Intel Arc B580 12GB*

## Installation

### Method 1: ComfyUI Manager (Recommended)

Open ComfyUI Manager → Custom Nodes → Install via Git URL, enter:

```
https://github.com/allanmeng/ComfyUI-Aila-XPU
```

After installation, you also need to download runtime files (first-time setup):
1. Go to the [Release page](https://github.com/allanmeng/ComfyUI-Aila-XPU/releases) and download `aila_runtime_dlls.zip`
2. Extract to `ComfyUI/custom_nodes/ComfyUI-Aila-XPU/aila_runtime/`

### Method 2: Source Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/allanmeng/ComfyUI-Aila-XPU
cd ComfyUI-Aila-XPU
pip install -r requirements.txt
```

After installation, download runtime DLLs as described above.

### Method 3: Cloud Download

[https://pan.quark.cn/s/c793f4fbb990](https://pan.quark.cn/s/c793f4fbb990)

### Directory Structure (v0.1.7+)

Engine v0.1.7 adopts process-isolated architecture. `AilaShared.dll` is a lightweight C API proxy, actual inference runs in `AilaWorker.exe`:

```
ComfyUI-Aila-XPU/
├── AilaShared.dll              ← C API proxy (git tracked)
├── aila_runtime/
│   ├── AilaWorker.exe           ← Inference engine (git tracked)
│   ├── Aila.exe                 ← CLI tool
│   └── <oneAPI runtime DLLs>    ← Download from release
```

### Note

- Plugin code + `AilaShared.dll` + `AilaWorker.exe` are updated via `git pull`
- oneAPI runtime DLLs (~476 MB) are downloaded separately from Release, rarely need updating
<p style="color:red; font-size:16px; font-weight:bold">⚠ If your startup script sets <code>SYCL_CACHE_PERSISTENT=1</code>, comment it out (causes Aila worker crash)</p>

## Models

### Supported Models

| Model | Format | Use Case | Recommended | Location |
|:----|:----|:----|:----:|:---------|
| Qwen3.5-4B | [NF4](https://huggingface.co/Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision), [BF16](https://huggingface.co/Qwen/Qwen3.5-4B) | VLM captioning / LLM chat | **Recommended LLM (NF4)** | `models/aila/` |
| Qwen3.5-0.8B | [NF4](https://huggingface.co/Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision), [BF16](https://huggingface.co/Qwen/Qwen3.5-0.8B) | VLM captioning / LLM chat | Lightweight | `models/aila/` |
| huihui-Qwen3.5-4B-abliterated | [NF4](https://huggingface.co/huihui-ai/Qwen3.5-4B-abliterated), [BF16](https://huggingface.co/huihui-ai/Qwen3.5-4B-abliterated) | VLM captioning / LLM chat | Abliterated | `models/aila/` |
| Qwen3-4B | [NF4](https://huggingface.co/Blackwood416/Qwen3-4B-BNB-NF4), [BF16](https://huggingface.co/Qwen/Qwen3-4B) | LLM text only | Text-only inference | `models/aila/` |
| Qwen3-0.6B | [NF4](https://huggingface.co/Blackwood416/Qwen3-0.6B-BNB-NF4), [BF16](https://huggingface.co/Qwen/Qwen3-0.6B) | LLM text only | Lightweight test | `models/aila/` |
| Qwen3-ASR-1.7B | [NF4](https://huggingface.co/Blackwood416/Qwen3-ASR-1.7B-BNB-NF4), [BF16](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | ASR speech-to-text | **Recommended ASR (NF4)** | `models/aila/asr/` |
| Qwen3-ASR-0.6B | [NF4](https://huggingface.co/Blackwood416/Qwen3-ASR-0.6B-BNB-NF4), [BF16](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | ASR speech-to-text | Lightweight | `models/aila/asr/` |
| Qwen3-ForceAligner-0.6B | [NF4](https://huggingface.co/Blackwood416/Qwen3-ForceAligner-0.6B-BNB-NF4) | ASR forced alignment | Subtitle alignment | `models/aila/asr/` |
| Qwen3-TTS-12Hz-1.7B-Base | [BF16](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS / voice cloning | Better quality | `models/aila/tts/` |
| Qwen3-TTS-12Hz-0.6B-Base | [BF16](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | TTS / voice cloning | Lightweight | `models/aila/tts/` |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | [BF16](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | TTS with preset voices | 9 preset voices | `models/aila/tts/` |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | [BF16](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | TTS with preset voices | Lightweight presets | `models/aila/tts/` |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | [BF16](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | TTS voice design | Style-based voice gen | `models/aila/tts/` |

### Download Models

**Recommended: Download pre-exported NF4 models (ready to use):**

- [Blackwood416's HF Collection](https://huggingface.co/collections/Blackwood416/ailas-model-collections) — LLM + ASR NF4 models
- [Quark Cloud Drive](https://pan.quark.cn/s/5d795bb3c417) — Base model pack (some recommended models)

**Or export yourself:**

```bash
# LLM models
python export_model.py --from-hf Blackwood416/Qwen3.5-4B-BNB-NF4-with-vision
python export_model.py --from-hf Blackwood416/Qwen3.5-0.8B-BNB-NF4-with-vision

# ASR models (NF4 format)
python export_model.py --from-hf Blackwood416/Qwen3-ASR-1.7B-BNB-NF4
python export_model.py --from-hf Blackwood416/Qwen3-ASR-0.6B-BNB-NF4

# TTS models (BF16 format)
python export_model.py --from-hf Qwen/Qwen3-TTS-12Hz-1.7B-Base
python export_model.py --from-hf Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

### Model Storage Location

After downloading, place models in the corresponding directories:

| Purpose | Directory |
|:----|:-----|
| VLM captioning / LLM chat | `ComfyUI/models/aila/` or `ComfyUI/models/aila/llm/` |
| ASR speech-to-text | `ComfyUI/models/aila/asr/` |
| TTS text-to-speech | `ComfyUI/models/aila/tts/` |

## Usage

The plugin includes 6 nodes, organized into three groups:

### VLM Captioning / LLM Chat

| Node | Function |
|:----|:-----|
| `Aila LLM Loader (XPU)` | Load LLM/VLM model |
| `Aila LLM Captioner (XPU)` | Image captioning (with image) or text Q&A (without image) |

**Image captioning workflow:**
1. Add `Aila LLM Loader (XPU)` → Select model → Execute
2. Add `Aila LLM Captioner (XPU)` → Connect Loader's `MODEL` output
3. Connect image to `images` input
4. Configure `mode` (prompt/caption/danbooru), `user_prompt`, `system_prompt`, etc.
5. Execute → Output `TEXT`

**Text-only Q&A workflow:**
1. Add `Aila LLM Loader (XPU)` → Select a text-only model
2. Add `Aila LLM Captioner (XPU)` → Don't connect image input
3. Write question in `user_prompt`, role in `system_prompt`
4. Execute → Output `TEXT`

#### LLM Captioner Parameters

| Parameter | Description |
|:----|:-----|
| `mode` | Output mode: `prompt (SD prompt)`、`caption (description)`、`danbooru (Danbooru tags)` |
| `user_prompt` | Custom user instruction; leave empty for default |
| `system_prompt` | Custom system prompt; auto-selected based on mode if empty |
| `max_tokens` | Max generation tokens (default 256) |
| `temperature` | Sampling temperature (default 0.7) |
| `top_p` / `top_k` | Sampling parameters |
| `do_sample` | Enable sampling (disable for greedy decoding) |
| `seed` | Random seed (0=random) |
| `memory_cleanup` | GPU memory handling after generation |

**Three modes:**

| Mode | Language | Style | Use Case |
|:----:|:--------:|:---------|:---------|
| **prompt** | English | Full SD prompt | Copy directly to positive prompt |
| **caption** | Chinese | Detailed natural description | Image description, content logging |
| **danbooru** | English | Comma-separated Danbooru tags | Image tagging |

### ASR Speech-to-Text

| Node | Function |
|:----|:-----|
| `Aila ASR Loader (XPU)` | Load ASR model |
| `Aila ASR Transcriber (XPU)` | Speech-to-text |

**Workflow:**
1. Add `Aila ASR Loader (XPU)` → Select ASR model → Execute
2. Add `Aila ASR Transcriber (XPU)` → Connect Loader's `ASR_MODEL` output
3. Connect audio (WAV/MP3/etc.) to `audio` input
4. Execute → Output `TEXT`

#### ASR Transcriber Parameters

| Parameter | Default | Description |
|:----|:----:|:------|
| `forced_lang` | auto | Force audio language for better accuracy |
| `asr_system` | empty | Context hint (e.g. "This is a tech lecture") |
| `asr_segment` | -1 | Segment duration (seconds). -1/0=no segment, >0=segment. Long audio: 30~60s recommended |
| `max_tokens` | 1024 | Max transcription length |
| `seed` | 0 | Random seed |
| `memory_cleanup` | persistent | GPU memory handling |

### TTS Text-to-Speech

| Node | Function |
|:----|:-----|
| `Aila TTS Loader (XPU)` | Load TTS model |
| `Aila TTS Synthesizer (XPU)` | Text-to-speech |

**Workflow:**
1. Add `Aila TTS Loader (XPU)` → Select TTS model → Execute
2. Add `Aila TTS Synthesizer (XPU)` → Connect Loader's `TTS_MODEL` output
3. Enter text in the `text` input
4. Execute → Output `AUDIO`

#### TTS Synthesizer Parameters

| Parameter | Default | Description |
|:----|:----:|:------|
| `text` | Required | Text to synthesize |
| `max_new_tokens` | -1 | Max tokens per segment. -1=8192 (~19 min, effectively unlimited) |
| `auto_segment` | False | Split text by sentence and concatenate audio. Recommended for long text |
| `seed` | 0 | Random seed |
| `memory_cleanup` | persistent | GPU memory handling |
| `debug` | False | Debug logging |

## Technical Notes

- This plugin calls the Aila C API via **ctypes**, encoding images into vision tokens for caption generation
- Image processing: GPU tensor → temporary PNG → Aila API reads | Auto-cleaned after generation
- Supports single / batch image processing

## Credits

- [Aila](https://github.com/Blackwood416/Aila) — Intel Arc inference engine
- [Qwen](https://github.com/QwenLM/Qwen) — Alibaba Qwen model series
- Intel — XPU PyTorch + oneDNN acceleration
