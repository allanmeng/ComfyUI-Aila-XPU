"""Aila 提示词反推插件 - ComfyUI 节点定义"""

import json
import os
import sys
import ctypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

import folder_paths
from comfy_api.latest import io

from .utils import image_tensor_to_data_uri

# ─── 常量 ───────────────────────────────────────────────────────────────────

PLUGIN_DIR = Path(__file__).parent
CONFIG_FILE = PLUGIN_DIR / "config.json"
try:
    # 优先使用 ComfyUI 注册的 aila 文件夹路径
    _aila_paths, _ = folder_paths.folder_names_and_paths["aila"]
    AILA_MODEL_DIR = Path(_aila_paths[0])
except (KeyError, IndexError):
    # 回退到 models/aila
    AILA_MODEL_DIR = Path(folder_paths.base_path) / "models" / "aila"

# 创建 LLM/ASR/TTS 子目录（首次安装时自动建立）
AILA_LLM_MODEL_DIR = AILA_MODEL_DIR / "llm"
AILA_LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 临时图像目录
TEMP_DIR = PLUGIN_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)
_TEMP_COUNTER = 0


def _next_temp_id() -> int:
    """线程安全的临时文件 ID 生成。"""
    global _TEMP_COUNTER
    _TEMP_COUNTER += 1
    return _TEMP_COUNTER

# 默认 DLL 搜索路径
_DEFAULT_DLL_CANDIDATES = [
    PLUGIN_DIR / "AilaShared.dll",
    PLUGIN_DIR / "build" / "AilaShared.dll",
    Path(folder_paths.base_path) / ".." / "Aila" / "build" / "AilaShared.dll",
]

# ─── 全局引擎缓存 ────────────────────────────────────────────────────────────
# model_path -> {"engine": c_void_p, "lib": ctypes.CDLL, "max_seq_len": int}
_aila_engines: Dict[str, Dict[str, Any]] = {}


# ─── 配置管理 ────────────────────────────────────────────────────────────────

def load_config() -> Dict[str, Any]:
    """加载用户配置."""
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_dll_path() -> Optional[Path]:
    """获取 AilaShared.dll 路径，按优先级搜索。"""
    config = load_config()
    cfg_path = config.get("dll_path")
    if cfg_path:
        p = Path(cfg_path)
        if not p.is_absolute():
            p = PLUGIN_DIR / p  # 相对路径基于插件目录解析
        if p.exists():
            return p.resolve()

    for candidate in _DEFAULT_DLL_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    return None


def get_model_folders() -> List[Path]:
    """获取模型扫描目录。"""
    config = load_config()
    user_folders = [Path(f) for f in config.get("model_folders", []) if os.path.exists(f)]
    default_folders = [AILA_MODEL_DIR] if AILA_MODEL_DIR.exists() else []
    # 也扫描 aila/llm/ 子目录，兼容新路径
    if AILA_LLM_MODEL_DIR.exists():
        default_folders.append(AILA_LLM_MODEL_DIR)
    return default_folders + user_folders


def find_aila_models() -> List[str]:
    """扫描模型目录，返回可用的模型名称列表（目录名）。"""
    folders = get_model_folders()
    models = []
    seen = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue
            name = sub.name
            if name in seen:
                continue
            # Aila 模型目录需要包含 config.json
            if (sub / "config.json").exists():
                models.append(name)
                seen.add(name)
    return models


def find_model_path(model_name: str) -> Optional[Path]:
    """在模型目录中查找指定模型名的完整路径。"""
    for folder in get_model_folders():
        path = folder / model_name
        if path.is_dir():
            return path
    return None


# ─── Aila C API ctypes 绑定 ─────────────────────────────────────────────────

class AilaGenConfig(ctypes.Structure):
    """对应 Aila 的 AilaGenConfig 结构体。"""
    _fields_ = [
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("do_sample", ctypes.c_int),
        ("decode_chunk_size", ctypes.c_int),
        ("stream_chunk_size", ctypes.c_int),
    ]


class AilaAlignedWord(ctypes.Structure):
    """对应 Aila 的 AilaAlignedWord 结构体（强制对齐结果）。"""
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("start_ms", ctypes.c_int),
        ("end_ms", ctypes.c_int),
    ]


def _bind_functions(lib: ctypes.CDLL):
    """绑定 AilaShared.dll 中 C API 函数的调用签名。"""
    # 生命周期
    lib.aila_engine_create.restype = ctypes.c_void_p

    lib.aila_engine_init.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.aila_engine_init.restype = ctypes.c_int

    lib.aila_engine_destroy.argtypes = [ctypes.c_void_p]
    lib.aila_engine_destroy.restype = None

    # 生成
    lib.aila_generate_messages.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(AilaGenConfig),
    ]
    lib.aila_generate_messages.restype = ctypes.c_void_p

    # 内存管理
    lib.aila_free_string.argtypes = [ctypes.c_void_p]
    lib.aila_free_string.restype = None

    # 配置
    lib.aila_default_gen_config.restype = AilaGenConfig

    # 上下文
    lib.aila_engine_reset_context.argtypes = [ctypes.c_void_p]
    lib.aila_engine_reset_context.restype = None

    # 版本
    lib.aila_version.restype = ctypes.c_char_p

    # 错误
    lib.aila_last_error_code.argtypes = [ctypes.c_void_p]
    lib.aila_last_error_code.restype = ctypes.c_int
    lib.aila_last_error_message.argtypes = [ctypes.c_void_p]
    lib.aila_last_error_message.restype = ctypes.c_char_p

    # ASR 转录
    lib.aila_transcribe.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.c_char_p,                   # wav_path
        ctypes.POINTER(AilaGenConfig),     # config (可 NULL)
        ctypes.c_char_p,                   # forced_language (可 NULL)
        ctypes.c_char_p,                   # system_prompt (可 NULL)
        ctypes.c_float,                    # segment_sec
        ctypes.c_int,                      # past_text_conditioning
        ctypes.c_void_p,                   # token_callback (NULL)
        ctypes.c_void_p,                   # user_data (NULL)
        ctypes.POINTER(ctypes.c_char_p),   # language_out (输出)
    ]
    lib.aila_transcribe.restype = ctypes.c_void_p

    # TTS 合成
    lib.aila_synthesize_text_to_wav.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.c_char_p,                   # text
        ctypes.POINTER(ctypes.c_float),    # speaker_embedding (可 NULL)
        ctypes.c_int,                      # speaker_embedding_len
        ctypes.POINTER(AilaGenConfig),     # config (可 NULL)
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # out_samples (输出)
        ctypes.POINTER(ctypes.c_int),      # out_sample_count (输出)
    ]
    lib.aila_synthesize_text_to_wav.restype = ctypes.c_int

    # 音频采样释放
    lib.aila_free_samples.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.aila_free_samples.restype = None

    # 提取说话人嵌入（语音克隆用）
    lib.aila_extract_speaker_embedding.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.c_char_p,                   # audio_path
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # out_embedding (输出)
        ctypes.POINTER(ctypes.c_int),      # out_embedding_dim (输出)
    ]
    lib.aila_extract_speaker_embedding.restype = ctypes.c_int

    # 高级 TTS 合成（预设音色/风格/语言，写 WAV 文件）
    lib.aila_synthesize.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.c_char_p,                   # text
        ctypes.c_char_p,                   # reference_audio_path (可 NULL)
        ctypes.c_char_p,                   # speaker_name (可 NULL)
        ctypes.c_char_p,                   # instruct_text (可 NULL)
        ctypes.c_char_p,                   # language (可 NULL)
        ctypes.POINTER(AilaGenConfig),     # config (可 NULL)
        ctypes.c_char_p,                   # output_wav_path
    ]
    lib.aila_synthesize.restype = ctypes.c_int

    # 强制对齐（ForceAligner）
    lib.aila_align.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.POINTER(ctypes.c_float),    # audio_samples
        ctypes.c_int,                      # num_samples
        ctypes.c_int,                      # sample_rate
        ctypes.c_char_p,                   # text
        ctypes.c_char_p,                   # language (可 NULL)
        ctypes.POINTER(ctypes.POINTER(AilaAlignedWord)),  # out_words (输出)
        ctypes.POINTER(ctypes.c_int),      # out_count (输出)
    ]
    lib.aila_align.restype = ctypes.c_int

    lib.aila_align_words.argtypes = [
        ctypes.c_void_p,                   # engine
        ctypes.POINTER(ctypes.c_float),    # audio_samples
        ctypes.c_int,                      # num_samples
        ctypes.c_int,                      # sample_rate
        ctypes.POINTER(ctypes.c_char_p),   # words 数组
        ctypes.c_int,                      # num_words
        ctypes.POINTER(ctypes.POINTER(AilaAlignedWord)),  # out_words (输出)
        ctypes.POINTER(ctypes.c_int),      # out_count (输出)
    ]
    lib.aila_align_words.restype = ctypes.c_int

    lib.aila_free_aligned_words.argtypes = [
        ctypes.POINTER(AilaAlignedWord),   # words
        ctypes.c_int,                      # count
    ]
    lib.aila_free_aligned_words.restype = None


def load_aila_library(dll_path: Path) -> ctypes.CDLL:
    """加载 AilaShared.dll，返回 CDLL 实例。"""
    dll_dir = str(dll_path.parent.resolve())
    # 确保 DLL 所在目录在搜索路径中
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(dll_dir)
    else:
        # 备用：修改 PATH
        old_path = os.environ.get("PATH", "")
        if dll_dir not in old_path:
            os.environ["PATH"] = dll_dir + os.pathsep + old_path

    lib = ctypes.cdll.LoadLibrary(str(dll_path))
    _bind_functions(lib)
    return lib


def init_aila_engine(lib: ctypes.CDLL, model_path: str, max_seq_len: int = 4096) -> ctypes.c_void_p:
    """创建并初始化 Aila 引擎。返回 engine 指针。"""
    engine = lib.aila_engine_create()
    if not engine:
        raise RuntimeError("aila_engine_create 返回空指针")

    ret = lib.aila_engine_init(engine, model_path.encode("utf-8"), max_seq_len)
    if ret != 0:
        err_msg = _get_aila_error(lib, engine)
        lib.aila_engine_destroy(engine)
        raise RuntimeError(f"aila_engine_init 失败 (code={ret}): {err_msg}")

    return engine


def _get_aila_error(lib: ctypes.CDLL, engine: ctypes.c_void_p) -> str:
    """获取 Aila 引擎的最后一个错误消息（含错误码）。"""
    try:
        code = lib.aila_last_error_code(engine)
        msg_ptr = lib.aila_last_error_message(engine)
        msg = ctypes.string_at(msg_ptr).decode("utf-8", errors="replace") if msg_ptr else "无错误消息"
        return f"[code={code}] {msg}"
    except Exception as e:
        return f"获取错误时异常: {e}"


# ─── 系统提示词模板 ─────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "prompt": (
        "You are an expert at writing prompts for AI image generation. "
        "Describe the given image concisely in English as a high-quality prompt, "
        "covering: subject, pose, expression, clothing, background, lighting, "
        "art style, composition, color palette, and mood. "
        "Output only the prompt text, no explanations."
    ),
    "caption": (
        "Describe this image in detail in Chinese. "
        "Include the main subject, setting, actions, colors, composition, "
        "and any notable elements you observe."
    ),
    "danbooru": (
        "Generate Danbooru-style tags for this image. "
        "Output comma-separated tags in English. "
        "Include character names if recognizable, general tags for appearance, "
        "pose, expression, clothing, background, and art style. "
        "Output only tags, no explanations."
    ),
}


# ─── 全局引擎管理 ───────────────────────────────────────────────────────────

def _shutdown_engine(model_path: str):
    """销毁指定模型的引擎。"""
    entry = _aila_engines.pop(model_path, None)
    if entry:
        try:
            entry["lib"].aila_engine_destroy(entry["engine"])
        except Exception:
            pass


def _shutdown_all_engines():
    """销毁所有缓存的引擎。"""
    for model_path in list(_aila_engines.keys()):
        _shutdown_engine(model_path)


def _get_or_create_engine(
    dll_path: Path, model_path_str: str, max_seq_len: int = 4096,
) -> ctypes.c_void_p:
    """获取或创建引擎实例（缓存）。"""
    # 如果已存在且路径相同，直接返回
    existing = _aila_engines.get(model_path_str)
    if existing is not None:
        return existing["engine"]

    # 如果缓存的是其他模型，先清理
    _shutdown_all_engines()

    # 加载 DLL
    lib = load_aila_library(dll_path)
    engine = init_aila_engine(lib, model_path_str, max_seq_len)

    _aila_engines[model_path_str] = {
        "engine": engine,
        "lib": lib,
        "max_seq_len": max_seq_len,
    }
    return engine


# ─── 节点：AilaModelLoader ──────────────────────────────────────────────────

class AilaModelLoader(io.ComfyNode):
    """加载 Aila 引擎与模型。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = find_aila_models()
        if not models:
            models = ["<未找到模型，请将模型放入 models/aila/>"]

        return io.Schema(
            node_id="AilaModelLoader",
            display_name="Aila LLM Loader (XPU)",
            category="Aila",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=models,
                    tooltip="选择 Aila 模型（子目录中有 config.json 的目录）",
                ),
                io.Int.Input(
                    "max_seq_len",
                    default=4096,
                    min=512,
                    max=32768,
                    tooltip="最大序列长度",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("AILA_MODEL").Output(display_name="MODEL"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, max_seq_len: int = 4096) -> io.NodeOutput:
        try:
            # 查找模型路径
            model_path = find_model_path(model_name)
            if model_path is None:
                raise FileNotFoundError(
                    f"找不到模型目录: {model_name}。"
                    f" 请将模型放入 {AILA_MODEL_DIR}"
                )

            # 查找 DLL
            dll_path = get_dll_path()
            if dll_path is None:
                raise FileNotFoundError(
                    "找不到 AilaShared.dll。"
                    " 请确保已下载 Aila 发行版，"
                    " 并在 config.json 中设置 dll_path。"
                )

            # 获取或创建引擎
            engine = _get_or_create_engine(dll_path, str(model_path), max_seq_len)

            return io.NodeOutput({
                "model_name": model_name,
                "model_path": str(model_path),
                "dll_path": str(dll_path),
                "max_seq_len": max_seq_len,
            })

        except Exception as e:
            raise RuntimeError(f"AilaModelLoader 失败: {e}")


# ─── 节点：AilaCaptioner ────────────────────────────────────────────────────

class AilaCaptioner(io.ComfyNode):
    """基于 Aila 引擎的图像描述/提示词反推。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AilaCaptioner",
            display_name="Aila LLM Captioner (XPU)",
            category="Aila",
            inputs=[
                io.Custom("AILA_MODEL").Input(
                    "aila_model",
                    tooltip="来自 Aila LLM Loader 的模型",
                ),
                io.Image.Input(
                    "images",
                    tooltip="输入图像（不接则纯文本模式）",
                    optional=True,
                ),
                io.Combo.Input(
                    "mode",
                    options=[
                        "prompt (SD提示词)",
                        "caption (图片描述)",
                        "danbooru (Danbooru标签)",
                    ],
                    default="prompt (SD提示词)",
                    tooltip="反推模式: prompt=SD提示词, caption=详细描述, danbooru=标签",
                ),
                io.String.Input(
                    "user_prompt",
                    multiline=True,
                    default="Describe this image in detail.",
                    tooltip="发送给模型的用户消息",
                    optional=True,
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="",
                    tooltip="自定义 system prompt（留空则使用模式默认）",
                    optional=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=256,
                    min=16,
                    max=8192,
                    tooltip="最大生成 token 数",
                    optional=True,
                ),
                io.Float.Input(
                    "temperature",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip="采样温度",
                    optional=True,
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Top-P 采样",
                    optional=True,
                ),
                io.Int.Input(
                    "top_k",
                    default=40,
                    min=1,
                    max=200,
                    tooltip="Top-K 采样",
                    optional=True,
                ),
                io.Boolean.Input(
                    "do_sample",
                    default=True,
                    tooltip="启用采样（关闭则为贪心解码）",
                    optional=True,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="随机种子（0=随机）",
                    optional=True,
                ),
                io.Combo.Input(
                    "memory_cleanup",
                    options=[
                        "persistent (不释放)",
                        "full_cleanup (释放显存, 清理缓存)",
                    ],
                    default="persistent (不释放)",
                    tooltip="生成后显存清理方式",
                    optional=True,
                ),
                io.Float.Input(
                    "rep_penalty",
                    default=1.0,
                    min=1.0,
                    max=5.0,
                    step=0.01,
                    tooltip="重复惩罚系数（>1.0 降低重复，推荐 1.0~1.2）",
                    optional=True,
                ),
                io.Float.Input(
                    "pres_penalty",
                    default=0.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip="存在惩罚（>0 鼓励新话题，推荐 0.0~0.1）",
                    optional=True,
                ),
                io.Float.Input(
                    "freq_penalty",
                    default=0.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip="频率惩罚（>0 降低高频词，推荐 0.0~0.1）",
                    optional=True,
                ),
                io.Boolean.Input(
                    "enable_thinking",
                    default=True,
                    tooltip="启用 think 思维链输出（关闭后抑制 <think> 内容）",
                    optional=True,
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    tooltip="启用调试日志（显示 Token ID 等详细信息）",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="TEXT"),
            ],
        )

    @classmethod
    def execute(
        cls,
        aila_model: Dict[str, Any],
        images: Optional[torch.Tensor] = None,
        mode: str = "prompt",
        user_prompt: str = "Describe this image in detail.",
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        do_sample: bool = True,
        seed: int = 0,
        memory_cleanup: str = "persistent (不释放)",
        rep_penalty: float = 1.0,
        pres_penalty: float = 0.0,
        freq_penalty: float = 0.0,
        enable_thinking: bool = True,
        debug: bool = False,
    ) -> io.NodeOutput:
        """执行单张或批量图像反推。"""
        try:
            # 校验输入
            if not isinstance(aila_model, dict) or "model_path" not in aila_model:
                raise ValueError("无效的模型数据，请从 Aila LLM Loader 连接")

            model_path = aila_model["model_path"]
            engine_entry = _aila_engines.get(model_path)
            if engine_entry is None:
                # 模型未加载，可能是被 full_cleanup 销毁了，自动重载
                dll_path = get_dll_path()
                if dll_path is None:
                    raise RuntimeError("找不到 AilaShared.dll，请检查配置")
                _get_or_create_engine(dll_path, model_path)
                engine_entry = _aila_engines.get(model_path)
                if engine_entry is None:
                    raise RuntimeError(f"自动重载模型 '{model_path}' 失败")

            lib = engine_entry["lib"]
            engine = engine_entry["engine"]

            # 确定 system prompt（从带中文描述的 mode 中提取 key）
            mode_key = mode.split(" (")[0] if " (" in mode else mode
            sp = system_prompt.strip() or SYSTEM_PROMPTS.get(mode_key, SYSTEM_PROMPTS["prompt"])

            # 生成参数（在 reset 之前创建 config）
            cfg = lib.aila_default_gen_config()
            cfg.max_new_tokens = max_tokens
            cfg.temperature = temperature
            cfg.top_k = top_k
            cfg.top_p = top_p
            cfg.do_sample = 1 if do_sample else 0
            cfg.repetition_penalty = rep_penalty
            cfg.presence_penalty = pres_penalty
            cfg.frequency_penalty = freq_penalty
            if seed > 0:
                pass

            # 调试日志
            if debug:
                os.environ["AILA_DEBUG_TOKEN_IDS"] = "1"
            elif "AILA_DEBUG_TOKEN_IDS" in os.environ:
                del os.environ["AILA_DEBUG_TOKEN_IDS"]

            # Think 模式控制
            final_user_prompt = user_prompt
            if not enable_thinking:
                final_user_prompt = user_prompt.strip() + " /no_think" if user_prompt else "/no_think"

            results = []
            tmp_files = []

            if images is not None:
                # 有图模式：逐张处理（单张或批量）
                B = images.shape[0]
                for i in range(B):
                    global _TEMP_COUNTER
                    _TEMP_COUNTER += 1
                    img_tensor = images[i]

                    arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
                    if arr.shape[2] == 1:
                        arr = arr.repeat(3, axis=2)
                    pil_img = Image.fromarray(arr)
                    tmp_path = str(TEMP_DIR / f"aila_img_{_TEMP_COUNTER:06d}.png")
                    pil_img.save(tmp_path, format="PNG")
                tmp_files.append(tmp_path)

                # 构建 messages（有图模式）
                content: list[dict] = [
                    {"type": "image", "image": tmp_path},
                    {"type": "text", "text": final_user_prompt},
                ]

                messages = [
                    {"role": "system", "content": sp},
                    {"role": "user", "content": content},
                ]

                messages_json = json.dumps(messages, ensure_ascii=False)
                print(f"[Aila Engine] Image path: {tmp_path}")
                print(f"[Aila Engine] System: {sp[:100]}")
                print(f"[Aila Engine] User: {final_user_prompt[:100]}")

                # 调用 Aila C API
                result_ptr = lib.aila_generate_messages(
                    engine,
                    messages_json.encode("utf-8"),
                    ctypes.byref(cfg),
                )

                if not result_ptr:
                    err = _get_aila_error(lib, engine)
                    print(f"[Aila Captioner] ERROR: {err}")
                    raise RuntimeError(f"Aila 推理失败: {err}")

                # 读取结果
                raw_bytes = ctypes.string_at(result_ptr)
                text = raw_bytes.decode("utf-8", errors="replace")
                print(f"[Aila Captioner] Raw bytes ({len(raw_bytes)}): {raw_bytes[:200]}")
                print(f"[Aila Captioner] Decoded: {text[:200]}")
                lib.aila_free_string(result_ptr)

                results.append(text)

            else:
                # 纯文本模式（不接图片）
                messages = [
                    {"role": "system", "content": sp},
                    {"role": "user", "content": final_user_prompt},
                ]

                messages_json = json.dumps(messages, ensure_ascii=False)
                print(f"[Aila Engine] System: {sp[:100]}")
                print(f"[Aila Engine] User: {final_user_prompt[:100]}")

                result_ptr = lib.aila_generate_messages(
                    engine,
                    messages_json.encode("utf-8"),
                    ctypes.byref(cfg),
                )

                if not result_ptr:
                    err = _get_aila_error(lib, engine)
                    print(f"[Aila Captioner] ERROR: {err}")
                    raise RuntimeError(f"Aila 推理失败: {err}")

                raw_bytes = ctypes.string_at(result_ptr)
                text = raw_bytes.decode("utf-8", errors="replace")
                print(f"[Aila Captioner] Raw bytes ({len(raw_bytes)}): {raw_bytes[:200]}")
                print(f"[Aila Captioner] Decoded: {text[:200]}")
                lib.aila_free_string(result_ptr)

                results.append(text)

            # 清理临时文件
            for f in tmp_files:
                try:
                    os.unlink(f)
                except Exception:
                    pass

            # 单张直接返回文本，多张用 \n---\n 分隔
            output = results[0] if len(results) == 1 else "\n---\n".join(
                f"[{i}] {t}" for i, t in enumerate(results)
            )

            # 可选：生成后释放 GPU 显存
            cleanup_key = memory_cleanup.split(" (")[0] if " (" in memory_cleanup else memory_cleanup
            if cleanup_key == "full_cleanup":
                print(f"[Aila Engine] 彻底清理: {model_path} + XPU 缓存")
                _shutdown_engine(model_path)
                try:
                    if hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.empty_cache()
                        torch.xpu.synchronize()
                except Exception:
                    pass

            return io.NodeOutput(output)

        except Exception as e:
            raise RuntimeError(f"AilaCaptioner 失败: {e}")


# ─── 插件卸载时的引擎清理 ───────────────────────────────────────────────────

import atexit
atexit.register(_shutdown_all_engines)


# ─── ASR 模型目录 ─────────────────────────────────────────────────────────

AILA_ASR_MODEL_DIR = AILA_MODEL_DIR / "asr"
AILA_ASR_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def find_aila_asr_models() -> List[str]:
    """扫描 models/aila/asr/ 目录，返回可用的 ASR 模型名称列表。"""
    models = []
    if not AILA_ASR_MODEL_DIR.is_dir():
        return models
    for sub in sorted(AILA_ASR_MODEL_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "config.json").exists():
            models.append(sub.name)
    return models


def _find_force_aligner_models() -> List[str]:
    """扫描 models/aila/asr/ 目录，返回可用的 ForceAligner 模型列表。"""
    models = []
    if not AILA_ASR_MODEL_DIR.is_dir():
        return ["<未找到 ForceAligner 模型>"]
    for sub in sorted(AILA_ASR_MODEL_DIR.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name.lower()
        if "forcealigner" in name.replace("_", "").replace("-", "") and (sub / "config.json").exists():
            models.append(sub.name)
    return models if models else ["<未找到 ForceAligner 模型>"]


# ─── 语音转 WAV 辅助 ───────────────────────────────────────────────────────

import soundfile as sf
import numpy as np


def _save_audio_to_wav(waveform: torch.Tensor, sample_rate: int, output_path: str):
    """将 ComfyUI AUDIO tensor 保存为临时 WAV 文件。

    Args:
        waveform: shape (batch, channels, samples) 或 (1, channels, samples)
        sample_rate: 采样率
        output_path: 输出 WAV 文件路径
    """
    wav = waveform[0]                     # (channels, samples)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0)      # 多声道 → 单声道
    else:
        wav = wav.squeeze(0)              # (samples,)
    np_wav = wav.cpu().numpy().astype(np.float32)
    sf.write(output_path, np_wav, sample_rate, format="WAV")


ASR_SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian",
]


# ─── 节点：AilaASRLoader ───────────────────────────────────────────────────

class AilaASRLoader(io.ComfyNode):
    """加载 Aila ASR 语音识别模型。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = find_aila_asr_models()
        if not models:
            models = ["<未找到 ASR 模型，请将模型放入 models/aila/asr/>"]

        return io.Schema(
            node_id="AilaASRLoader",
            display_name="Aila ASR Loader (XPU)",
            category="Aila",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=models,
                    tooltip="选择 ASR 模型（目录中有 config.json）",
                ),
                io.Int.Input(
                    "max_seq_len",
                    default=4096,
                    min=512,
                    max=32768,
                    tooltip="最大序列长度。ASR 场景 2048 足够，不分段长录音时需加大",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("AILA_ASR_MODEL").Output(display_name="ASR_MODEL"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, max_seq_len: int = 4096) -> io.NodeOutput:
        try:
            model_path = AILA_ASR_MODEL_DIR / model_name
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"找不到 ASR 模型目录: {model_name}。"
                    f" 请将模型放入 {AILA_ASR_MODEL_DIR}"
                )

            dll_path = get_dll_path()
            if dll_path is None:
                raise FileNotFoundError(
                    "找不到 AilaShared.dll。"
                    " 请确保已下载 Aila 发行版并配置 config.json"
                )

            engine = _get_or_create_engine(dll_path, str(model_path), max_seq_len)

            return io.NodeOutput({
                "model_name": model_name,
                "model_path": str(model_path),
                "dll_path": str(dll_path),
                "max_seq_len": max_seq_len,
            })

        except Exception as e:
            raise RuntimeError(f"AilaASRLoader 失败: {e}")


# ─── 节点：AilaTranscriber ─────────────────────────────────────────────────

class AilaTranscriber(io.ComfyNode):
    """基于 Aila ASR 模型的语音转文字。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AilaTranscriber",
            display_name="Aila ASR Transcriber (XPU)",
            category="Aila",
            inputs=[
                io.Custom("AILA_ASR_MODEL").Input(
                    "aila_model",
                    tooltip="来自 Aila ASR Loader 的模型",
                ),
                io.Audio.Input(
                    "audio",
                    tooltip="输入音频 (WAV)",
                    optional=False,
                ),
                io.Combo.Input(
                    "forced_lang",
                    options=ASR_SUPPORTED_LANGUAGES,
                    default="auto",
                    tooltip="强制指定音频语言，提高转录准确率。设为 auto 让模型自动检测",
                    optional=True,
                ),
                io.String.Input(
                    "asr_system",
                    multiline=True,
                    default="",
                    tooltip="转录上下文提示，让模型更准确识别特定场景的词。比如：这是一个关于计算机技术的讲座",
                    optional=True,
                ),
                io.Float.Input(
                    "asr_segment",
                    default=-1.0,
                    min=-1.0,
                    max=300.0,
                    step=1.0,
                    tooltip="音频分段处理时长（秒）。-1或0=不分段，>0按此秒数分段如30。长音频推荐30~60秒",
                    optional=True,
                ),
                io.Int.Input(
                    "max_tokens",
                    default=1024,
                    min=64,
                    max=8192,
                    tooltip="最大转录长度。10秒音频约需256，1分钟约512，6分钟约2048",
                    optional=True,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="随机种子。0=每次结果可能不同；固定值可复现相同转录结果",
                    optional=True,
                ),
                io.Combo.Input(
                    "memory_cleanup",
                    options=[
                        "persistent (不释放)",
                        "full_cleanup (释放显存, 清理缓存)",
                    ],
                    default="persistent (不释放)",
                    tooltip="转录完成后显存处理方式。persistent=继续占用加速连续转录；full_cleanup=释放给其他模型用",
                    optional=True,
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    tooltip="开启后控制台显示引擎详细日志和 Token ID 信息，排查问题用",
                    optional=True,
                ),
                # ── 强制对齐参数 ──
                io.Combo.Input(
                    "force_aligner_model",
                    options=["None (不加载)"] + [m for m in _find_force_aligner_models() if not m.startswith("<")],
                    default="None (不加载)",
                    tooltip="选择 ForceAligner 模型，选中后额外输出带时间戳的 SRT 字幕",
                    optional=True,
                ),
                io.Combo.Input(
                    "subtitle_mode",
                    options=["按断句", "按词", "按字"],
                    default="按断句",
                    tooltip="字幕拆分方式：按断句（推荐，按句号/长间隔拆分）、按词（jieba 分词）、按字（逐字时间戳）。仅启用了 ForceAligner 模型时有效",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="TEXT"),
                io.String.Output(display_name="SUBTITLES"),
            ],
        )

    @classmethod
    def execute(
        cls,
        aila_model: Dict[str, Any],
        audio: Dict[str, Any],
        forced_lang: str = "auto",
        asr_system: str = "",
        asr_segment: float = -1.0,
        max_tokens: int = 1024,
        seed: int = 0,
        memory_cleanup: str = "persistent (不释放)",
        debug: bool = False,
        force_aligner_model: str = None,
        subtitle_mode: str = "按断句",
    ) -> io.NodeOutput:
        """执行语音转文字。"""
        tmp_path = None
        try:
            # 校验输入
            if not isinstance(aila_model, dict) or "model_path" not in aila_model:
                raise ValueError("无效的模型数据，请从 Aila ASR Loader 连接")

            if audio is None:
                raise ValueError("请连接音频输入")

            model_path = aila_model["model_path"]
            engine_entry = _aila_engines.get(model_path)
            if engine_entry is None:
                dll_path = get_dll_path()
                if dll_path is None:
                    raise RuntimeError("找不到 AilaShared.dll，请检查配置")
                _get_or_create_engine(dll_path, model_path)
                engine_entry = _aila_engines.get(model_path)
                if engine_entry is None:
                    raise RuntimeError(f"自动重载模型 '{model_path}' 失败")

            lib = engine_entry["lib"]
            engine = engine_entry["engine"]

            # 调试日志
            if debug:
                os.environ["AILA_DEBUG_TOKEN_IDS"] = "1"
            elif "AILA_DEBUG_TOKEN_IDS" in os.environ:
                del os.environ["AILA_DEBUG_TOKEN_IDS"]

            # 提取音频数据并保存为临时 WAV
            waveform = audio["waveform"]       # (batch, channels, samples)
            sample_rate = audio["sample_rate"]  # int

            global _TEMP_COUNTER
            _TEMP_COUNTER += 1
            tmp_path = str(TEMP_DIR / f"aila_asr_{_TEMP_COUNTER:06d}.wav")
            _save_audio_to_wav(waveform, sample_rate, tmp_path)
            print(f"[Aila Transcriber] Audio saved: {tmp_path} (sr={sample_rate})")

            # 生成参数
            cfg = lib.aila_default_gen_config()
            cfg.max_new_tokens = max_tokens
            cfg.temperature = 0.0  # ASR 推荐贪心解码
            cfg.do_sample = 0

            # 语言参数
            forced_lang_c = forced_lang.encode("utf-8") if forced_lang and forced_lang != "auto" else None
            asr_system_c = asr_system.encode("utf-8") if asr_system.strip() else "请准确转录音频内容。".encode("utf-8")

            # 接收检测到的语言
            lang_ptr = ctypes.c_char_p()

            # 分段：-1或0=不分段，>0按秒分段
            seg_sec = max(0.0, asr_segment)

            result_ptr = lib.aila_transcribe(
                engine,
                tmp_path.encode("utf-8"),
                ctypes.byref(cfg),
                forced_lang_c,
                asr_system_c,
                ctypes.c_float(seg_sec),
                0,  # past_text_conditioning (已移除，统一关闭)
                None,  # token_callback
                None,  # user_data
                ctypes.byref(lang_ptr),
            )

            if not result_ptr:
                err = _get_aila_error(lib, engine)
                print(f"[Aila Transcriber] ERROR: {err}")
                raise RuntimeError(f"Aila ASR 推理失败: {err}")

            # 从原始指针读取字符串（c_void_p 需用 string_at）
            raw_bytes = ctypes.string_at(result_ptr)
            text = raw_bytes.decode("utf-8", errors="replace")
            detected_lang = lang_ptr.value.decode("utf-8", errors="replace") if lang_ptr.value else ""

            print(f"[Aila Transcriber] Result: {text[:200]}")
            if detected_lang:
                print(f"[Aila Transcriber] Detected language: {detected_lang}")
            lib.aila_free_string(result_ptr)
            if lang_ptr.value:
                lib.aila_free_string(lang_ptr)

            # ── 强制对齐（可选）──────────────────────────────────────────
            subtitles = ""
            if force_aligner_model and force_aligner_model not in ("None (不加载)", "<未找到 ForceAligner 模型>"):
                align_model_path = str(AILA_ASR_MODEL_DIR / force_aligner_model)
                print(f"[Aila Transcriber] Loading ForceAligner: {force_aligner_model}")

                # 加载 ForceAligner 引擎
                align_entry = _aila_engines.get(align_model_path)
                if align_entry is None:
                    dll_path = get_dll_path()
                    if dll_path is None:
                        raise RuntimeError("找不到 AilaShared.dll，请检查配置")
                    _get_or_create_engine(dll_path, align_model_path)
                    align_entry = _aila_engines.get(align_model_path)
                    if align_entry is None:
                        raise RuntimeError(f"加载 ForceAligner '{force_aligner_model}' 失败")

                align_lib = align_entry["lib"]
                align_engine = align_entry["engine"]

                # 从临时 WAV 文件读取音频 PCM 数据（重采样到 16kHz）
                import soundfile as sf
                audio_data, audio_sr = sf.read(tmp_path, dtype="float32")
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                import numpy as np
                audio_data = audio_data.flatten().astype(np.float32)

                # ForceAligner 需要 16kHz 音频
                target_sr = 16000
                if audio_sr != target_sr:
                    import numpy as np
                    old_len = len(audio_data)
                    new_len = int(old_len * target_sr / audio_sr)
                    audio_data = np.interp(
                        np.linspace(0, old_len - 1, new_len),
                        np.arange(old_len),
                        audio_data
                    ).astype(np.float32)
                    audio_sr = target_sr

                # 确保内存连续
                audio_data = np.ascontiguousarray(audio_data)

                # 转为 PCM float 数组
                lang = forced_lang.encode("utf-8") if forced_lang and forced_lang != "auto" else "Chinese".encode("utf-8")

                # 长音频分块对齐（每块 ~2 分钟，音频编码+文本合计不超 max_seq=4096）
                max_chunk_samples = 16000 * 120  # 2分钟 @ 16kHz
                total_samples = len(audio_data)
                total_text = text
                text_len = len(total_text)
                lang_str = forced_lang if forced_lang and forced_lang != "auto" else "Chinese"

                # 计算分块（直接按音频时长占比切文本）
                chunk_starts = list(range(0, total_samples, max_chunk_samples))
                all_subtitle_entries = []  # [(ch, start_ms, end_ms)]

                print(f"[Aila Transcriber] Splitting {total_samples/16000:.0f}s audio into {len(chunk_starts)} chunk(s) for ForceAligner")

                for ck, chunk_start in enumerate(chunk_starts):
                    chunk_end = min(chunk_start + max_chunk_samples, total_samples)
                    chunk_audio = audio_data[chunk_start:chunk_end]
                    chunk_samples = len(chunk_audio)

                    # 按比例切文本
                    text_start_ratio = chunk_start / total_samples
                    text_end_ratio = chunk_end / total_samples
                    chunk_text_start = int(text_len * text_start_ratio)
                    chunk_text_end = int(text_len * text_end_ratio)
                    chunk_text = total_text[chunk_text_start:chunk_text_end]
                    # 微调：往后扩展到最近的完整词边界
                    while chunk_text_end < text_len and total_text[chunk_text_end - 1] not in "，。！？、；：\n ":
                        chunk_text_end += 1
                    chunk_text = total_text[chunk_text_start:chunk_text_end]

                    print(f"[Aila Transcriber]  Chunk {ck+1}/{len(chunk_starts)}: audio [{chunk_start//16000}s-{chunk_end//16000}s], text[{chunk_text_start}:{chunk_text_end}]")

                    # 对齐当前块
                    chunk_audio_ptr = np.ascontiguousarray(chunk_audio).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    out_words_ptr = ctypes.POINTER(AilaAlignedWord)()
                    out_count = ctypes.c_int()
                    ret = align_lib.aila_align(
                        align_engine,
                        chunk_audio_ptr,
                        chunk_samples,
                        audio_sr,
                        chunk_text.encode("utf-8", errors="replace"),
                        lang,
                        ctypes.byref(out_words_ptr),
                        ctypes.byref(out_count),
                    )

                    if ret != 0:
                        err = _get_aila_error(align_lib, align_engine)
                        print(f"[Aila Transcriber]  Chunk {ck+1} align failed (code={ret}): {err}")
                        # 释放已分配的对齐结果
                        if out_words_ptr and out_count.value > 0:
                            align_lib.aila_free_aligned_words(out_words_ptr, out_count.value)
                        continue

                    # 收集当前块的逐字结果，时间偏移量 = chunk_start 对应的毫秒数
                    time_offset_ms = int(chunk_start / audio_sr * 1000)
                    for i in range(out_count.value):
                        w = out_words_ptr[i]
                        ch = ctypes.string_at(w.text).decode("utf-8", errors="replace") if w.text else ""
                        st = w.start_ms + time_offset_ms if w.start_ms >= 0 else -1
                        et = w.end_ms + time_offset_ms if w.end_ms > 0 else 0
                        all_subtitle_entries.append((ch, st, et))

                    # 释放对齐结果
                    align_lib.aila_free_aligned_words(out_words_ptr, out_count.value)

                if all_subtitle_entries:
                    import jieba
                    char_entries = all_subtitle_entries

                    # 2. 用对齐后的字符序列重建文本，jieba 分词
                    aligned_text = "".join(ch[0] for ch in char_entries if ch[0])
                    tokens = list(jieba.tokenize(aligned_text))
                    # tokens → [(word, char_start, char_end)]

                    # 3. 按词聚合时间
                    word_groups = []  # [(word_text, start_ms, end_ms)]
                    for word, c_start, c_end in tokens:
                        seg_chars = char_entries[c_start:c_end]
                        times = [(st, et) for _, st, et in seg_chars if st >= 0 and et > 0]
                        if times:
                            min_st = min(st for st, _ in times)
                            max_et = max(et for _, et in times)
                        else:
                            min_st = word_groups[-1][1] if word_groups else 0
                            max_et = word_groups[-1][2] if word_groups else 0
                        word_groups.append((word, min_st, max_et))

                    # 4. 根据 subtitle_mode 选择粒度
                    def _to_srt_time(ms):
                        h = ms // 3600000
                        m = (ms % 3600000) // 60000
                        s = (ms % 60000) // 1000
                        mill = ms % 1000
                        return f"{h:02d}:{m:02d}:{s:02d},{mill:03d}"

                    if subtitle_mode == "按字":
                        raw = [(ch, st, et) for ch, st, et in char_entries if ch and st >= 0 and et > 0]

                    elif subtitle_mode == "按词":
                        raw = [(w, st, et) for w, st, et in word_groups]

                    else:
                        # 按断句（默认）：用原始文本检测标点，用对齐结果获取时间
                        raw = []
                        cur_words, cur_start, cur_end = [], 0, 0
                        # char_offset = 累计了多少非标点字符
                        char_offset = 0
                        for i, (word, st, et) in enumerate(word_groups):
                            if not cur_words:
                                cur_start = st
                            cur_words.append(word)
                            cur_end = et
                            char_offset += len(word)
                            # 在原始文本中找到第 char_offset 个非标点字符的位置
                            orig_pos = 0
                            non_punct = 0
                            while orig_pos < len(text) and non_punct < char_offset:
                                if text[orig_pos] not in "，。！？、；：\n ":
                                    non_punct += 1
                                if non_punct < char_offset:
                                    orig_pos += 1
                            # 检查原始文本中该位置之后是否有标点
                            punct_after = ""
                            scan = orig_pos + 1
                            while scan < len(text) and text[scan] in "，。！？、；：\n ":
                                punct_after += text[scan]
                                scan += 1
                            cur_text = "".join(cur_words)
                            is_end = "。" in punct_after or "！" in punct_after or "？" in punct_after
                            is_soft = "，" in punct_after and len(cur_text) >= 8
                            gap = (word_groups[i + 1][1] - et) > 1500 if i + 1 < len(word_groups) else False
                            if is_end or gap:
                                raw.append((cur_text, cur_start, cur_end))
                                cur_words = []
                            elif is_soft:
                                raw.append((cur_text, cur_start, cur_end))
                                cur_words = []
                            elif len(cur_text) > 20:
                                raw.append((cur_text, cur_start, cur_end))
                                cur_words = []
                        if cur_words:
                            raw.append(("".join(cur_words), cur_start, cur_end))

                    print(f"[Aila Transcriber] subtitle_mode={subtitle_mode!r}, raw={len(raw)} entries")
                    lines = []
                    for idx, (txt, st, et) in enumerate(raw):
                        lines.append(f"{idx+1}")
                        lines.append(f"{_to_srt_time(st)} --> {_to_srt_time(et)}")
                        lines.append(txt)
                        lines.append("")
                    subtitles = "\n".join(lines)

            # 清理临时文件
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            # 显存清理
            cleanup_key = memory_cleanup.split(" (")[0] if " (" in memory_cleanup else memory_cleanup
            if cleanup_key == "full_cleanup":
                print(f"[Aila Transcriber] 彻底清理: {model_path} + XPU 缓存")
                _shutdown_engine(model_path)
                try:
                    if hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.empty_cache()
                        torch.xpu.synchronize()
                except Exception:
                    pass

            return (text, subtitles)

        except Exception as e:
            # 清理临时文件
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise RuntimeError(f"AilaTranscriber 失败: {e}")


# ─── TTS 模型目录 ─────────────────────────────────────────────────────────

TTS_MODEL_DIR = AILA_MODEL_DIR / "tts"
TTS_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def find_aila_tts_models() -> List[str]:
    """扫描 models/aila/tts/ 目录，返回可用的 TTS 模型名称列表。"""
    models = []
    if not TTS_MODEL_DIR.is_dir():
        return models
    for sub in sorted(TTS_MODEL_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "config.json").exists():
            models.append(sub.name)
    return models


# ─── 节点：AilaTTSLoader ───────────────────────────────────────────────────

class AilaTTSLoader(io.ComfyNode):
    """加载 Aila TTS 语音合成模型。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = find_aila_tts_models()
        if not models:
            models = ["<未找到 TTS 模型，请将模型放入 models/aila/tts/>"]

        return io.Schema(
            node_id="AilaTTSLoader",
            display_name="Aila TTS Loader (XPU)",
            category="Aila",
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=models,
                    tooltip="选择 TTS 模型（目录中有 config.json）",
                ),
                io.Int.Input(
                    "max_seq_len",
                    default=4096,
                    min=512,
                    max=32768,
                    tooltip="最大序列长度",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("AILA_TTS_MODEL").Output(display_name="TTS_MODEL"),
            ],
        )

    @classmethod
    def execute(cls, model_name: str, max_seq_len: int = 4096) -> io.NodeOutput:
        try:
            model_path = TTS_MODEL_DIR / model_name
            if not model_path.is_dir():
                raise FileNotFoundError(
                    f"找不到 TTS 模型目录: {model_name}。"
                    f" 请将模型放入 {TTS_MODEL_DIR}"
                )

            dll_path = get_dll_path()
            if dll_path is None:
                raise FileNotFoundError(
                    "找不到 AilaShared.dll。"
                    " 请确保已下载 Aila 发行版并配置 config.json"
                )

            engine = _get_or_create_engine(dll_path, str(model_path), max_seq_len)

            return io.NodeOutput({
                "model_name": model_name,
                "model_path": str(model_path),
                "dll_path": str(dll_path),
                "max_seq_len": max_seq_len,
            })

        except Exception as e:
            raise RuntimeError(f"AilaTTSLoader 失败: {e}")


# ─── 节点：AilaSynthesizer ─────────────────────────────────────────────────

TTS_SAMPLE_RATE = 24000  # Qwen3-TTS 输出采样率


class AilaSynthesizer(io.ComfyNode):
    """基于 Aila TTS 模型的文字转语音。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AilaSynthesizer",
            display_name="Aila TTS Synthesizer (XPU)",
            category="Aila",
            inputs=[
                io.Custom("AILA_TTS_MODEL").Input(
                    "aila_model",
                    tooltip="来自 Aila TTS Loader 的模型",
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    tooltip="要合成语音的文本内容",
                ),
                # ── 共用参数 ──
                io.Int.Input(
                    "max_new_tokens",
                    default=-1,
                    min=-1,
                    max=8192,
                    tooltip="最大生成 token 数，控制音频长度。设 -1 则设 8192（约19分钟，基本不限）；一段10秒语音约需256 tokens。配合 auto_segment 使用效果更好",
                    optional=True,
                ),
                io.Boolean.Input(
                    "auto_segment",
                    default=False,
                    tooltip="启用后按句号分句分段合成长文本，再拼接为完整音频。推荐长文本时开启",
                    optional=True,
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    control_after_generate=True,
                    tooltip="随机种子（0=随机）",
                    optional=True,
                ),
                io.Combo.Input(
                    "memory_cleanup",
                    options=[
                        "persistent (不释放)",
                        "full_cleanup (释放显存, 清理缓存)",
                    ],
                    default="persistent (不释放)",
                    tooltip="合成完成后显存清理方式",
                    optional=True,
                ),
                io.Boolean.Input(
                    "debug",
                    default=False,
                    tooltip="启用调试日志",
                    optional=True,
                ),
                # ── CustomVoice 专用参数 ──
                io.String.Input(
                    "_cv_divider",
                    display_name="分隔符",
                    default="───── CustomVoice 专用 ─────",
                    socketless=True,
                    optional=True,
                ),
                io.Combo.Input(
                    "speaker_name",
                    options=[
                        "默认",
                        "Ryan",
                        "Vivian",
                        "Aiden",
                        "Dylan",
                        "Eric",
                        "Ono_anna",
                        "Serena",
                        "Sohee",
                        "Uncle_fu",
                    ],
                    default="默认",
                    tooltip="CustomVoice 预设音色。选 \"默认\" 使用模型默认音色，或选预设名使用特定音色",
                    optional=True,
                ),
                io.Audio.Input(
                    "ref_audio",
                    tooltip="参考音频，用于 CustomVoice 语音克隆。提供后引擎会模仿该音频的音色",
                    optional=True,
                ),
                # ── VoiceDesign 专用参数 ──
                io.String.Input(
                    "_vd_divider",
                    display_name="分隔符",
                    default="───── VoiceDesign 专用 ─────",
                    socketless=True,
                    optional=True,
                ),
                io.String.Input(
                    "instruct",
                    multiline=True,
                    default="",
                    tooltip="VoiceDesign 风格指令，例如：\"温柔的女声，语速缓慢\"。仅 VoiceDesign 模型支持，Base/CustomVoice 会忽略",
                    optional=True,
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="AUDIO"),
            ],
        )

    @classmethod
    def execute(
        cls,
        aila_model: Dict[str, Any],
        text: str = "",
        max_new_tokens: int = -1,
        auto_segment: bool = False,
        seed: int = 0,
        memory_cleanup: str = "persistent (不释放)",
        debug: bool = False,
        speaker_name: str = "默认",
        ref_audio: Optional[Dict[str, Any]] = None,
        instruct: str = "",
        _cv_divider: str = "",
        _vd_divider: str = "",
    ) -> io.NodeOutput:
        """执行文字转语音。"""
        try:
            if not isinstance(aila_model, dict) or "model_path" not in aila_model:
                raise ValueError("无效的模型数据，请从 Aila TTS Loader 连接")

            if not text.strip():
                raise ValueError("请输入要合成的文本")

            model_path = aila_model["model_path"]
            engine_entry = _aila_engines.get(model_path)
            if engine_entry is None:
                dll_path = get_dll_path()
                if dll_path is None:
                    raise RuntimeError("找不到 AilaShared.dll，请检查配置")
                _get_or_create_engine(dll_path, model_path)
                engine_entry = _aila_engines.get(model_path)
                if engine_entry is None:
                    raise RuntimeError(f"自动重载模型 '{model_path}' 失败")

            lib = engine_entry["lib"]
            engine = engine_entry["engine"]

            if debug:
                os.environ["AILA_DEBUG_TOKEN_IDS"] = "1"
            elif "AILA_DEBUG_TOKEN_IDS" in os.environ:
                del os.environ["AILA_DEBUG_TOKEN_IDS"]

            # ── 预处理参考音频（语音克隆用）────────────────────────────────
            final_text = text.strip()
            print(f"[Aila Synthesizer] Text: {final_text[:100]} ({len(final_text)} chars)")

            ref_wav_path = None
            if ref_audio is not None and isinstance(ref_audio, dict):
                # 保存参考音频为临时 WAV
                ref_waveform = ref_audio.get("waveform")
                ref_sr = ref_audio.get("sample_rate", 24000)
                if ref_waveform is not None:
                    # 转为 float32 numpy
                    ref_np = ref_waveform.cpu().numpy().astype(np.float32)
                    # 合并多声道为单声道
                    while ref_np.ndim > 1:
                        ref_np = ref_np.mean(axis=0)
                    ref_np = ref_np.flatten()
                    # 确保采样率有效
                    if ref_sr is None or ref_sr <= 0:
                        ref_sr = 24000
                    ref_wav_path = str(TEMP_DIR / f"aila_ref_{_next_temp_id():04d}.wav")
                    import soundfile as sf
                    sf.write(ref_wav_path, ref_np, int(ref_sr), subtype="PCM_16")
                    print(f"[Aila Synthesizer] Ref audio saved: {ref_wav_path} ({len(ref_np)} samples, {ref_sr}Hz)")

            # ── 根据模型名称判断类型 ───────────────────────────────────────
            model_name = (aila_model.get("model_name", "") or "")
            model_name_lower = model_name.lower()
            model_type = "voicedesign" if "voicedesign" in model_name_lower else \
                         "customvoice" if "customvoice" in model_name_lower else \
                         "base"
            print(f"[Aila Synthesizer] Model: {model_name} → type={model_type}")

            # ── 选择合成路径 ──────────────────────────────────────────────
            # generate config: max_new_tokens=-1 传 8192 作为不限
            cfg = lib.aila_default_gen_config()
            cfg.max_new_tokens = max_new_tokens if max_new_tokens > 0 else 8192
            cfg_ptr = ctypes.byref(cfg)

            # 路径 B: 高级 API 模式（预设音色 / 风格指令）→ 走 aila_synthesize（写文件）
            has_spk = (speaker_name or "").strip() and (speaker_name or "").strip() != "默认"
            has_instruct = bool((instruct or "").strip())
            # 根据模型类型决定生效的参数
            if model_type == "customvoice":
                use_preset = has_spk
                effective_spk = speaker_name if has_spk else None
                effective_instruct = None
            elif model_type == "voicedesign":
                use_preset = has_instruct
                effective_spk = None
                effective_instruct = instruct if has_instruct else None
            else:  # base
                use_preset = False
                effective_spk = None
                effective_instruct = None

            if use_preset:
                # 分段
                if auto_segment:
                    for sep in ("。", "！", "？", "\n"):
                        final_text = final_text.replace(sep, sep + "\n")
                    sentences = [s.strip() for s in final_text.split("\n") if s.strip()]
                    seg_texts = []
                    current = ""
                    for s in sentences:
                        if len(current) + len(s) < 200 or not current:
                            current += s
                        else:
                            seg_texts.append(current)
                            current = s
                    if current:
                        seg_texts.append(current)
                    print(f"[Aila Synthesizer] Auto-segment: {len(seg_texts)} segment(s) (preset mode)")
                else:
                    seg_texts = [final_text]

                all_file_arrays: List[np.ndarray] = []
                for idx, seg_text in enumerate(seg_texts):
                    if len(seg_texts) > 1:
                        print(f"[Aila Synthesizer] Segment {idx+1}/{len(seg_texts)}: {seg_text[:60]}")

                    output_wav = str(TEMP_DIR / f"aila_tts_{_next_temp_id():04d}.wav")

                    has_spk = (speaker_name or "").strip() and (speaker_name or "").strip() != "默认"
                    has_instruct = bool((instruct or "").strip())

                    ret = lib.aila_synthesize(
                        engine,
                        seg_text.encode("utf-8"),
                        ref_wav_path.encode("utf-8") if ref_wav_path else None,
                        speaker_name.encode("utf-8") if effective_spk else None,
                        effective_instruct.encode("utf-8") if effective_instruct else None,
                        None,
                        cfg_ptr,
                        output_wav.encode("utf-8"),
                    )
                    if ret != 0:
                        err = _get_aila_error(lib, engine)
                        raise RuntimeError(f"Aila TTS 合成失败 (code={ret}): {err}")

                    import soundfile as sf
                    data, sr = sf.read(output_wav, dtype="float32")
                    all_file_arrays.append(data.flatten())
                    os.remove(output_wav)

                if ref_wav_path:
                    os.remove(ref_wav_path)

                full_array = np.concatenate(all_file_arrays) if len(all_file_arrays) > 1 else all_file_arrays[0]
                segments = seg_texts

            else:
                # 路径 A: 内存模式（默认 / 语音克隆）→ 走 aila_synthesize_text_to_wav
                speaker_embedding = None
                speaker_embedding_len = 0

                if ref_wav_path:
                    # 提取说话人嵌入
                    emb_ptr = ctypes.POINTER(ctypes.c_float)()
                    emb_dim = ctypes.c_int()
                    ret = lib.aila_extract_speaker_embedding(
                        engine,
                        ref_wav_path.encode("utf-8"),
                        ctypes.byref(emb_ptr),
                        ctypes.byref(emb_dim),
                    )
                    if ret != 0:
                        err = _get_aila_error(lib, engine)
                        raise RuntimeError(f"提取 speaker embedding 失败 (code={ret}): {err}")

                    speaker_embedding = emb_ptr
                    speaker_embedding_len = emb_dim.value
                    os.remove(ref_wav_path)
                    print(f"[Aila Synthesizer] Speaker embedding extracted: dim={speaker_embedding_len}")

                # 分段合成
                if auto_segment:
                    for sep in ("。", "！", "？", "\n"):
                        final_text = final_text.replace(sep, sep + "\n")
                    sentences = [s.strip() for s in final_text.split("\n") if s.strip()]
                    segments = []
                    current = ""
                    for s in sentences:
                        if len(current) + len(s) < 200 or not current:
                            current += s
                        else:
                            segments.append(current)
                            current = s
                    if current:
                        segments.append(current)
                    print(f"[Aila Synthesizer] Auto-segment: {len(segments)} segment(s)")
                else:
                    segments = [final_text]

                all_samples: List[np.ndarray] = []
                for idx, seg_text in enumerate(segments):
                    if len(segments) > 1:
                        print(f"[Aila Synthesizer] Segment {idx+1}/{len(segments)}: {seg_text[:60]}")

                    out_samples_ptr = ctypes.POINTER(ctypes.c_float)()
                    out_sample_count = ctypes.c_int()

                    ret = lib.aila_synthesize_text_to_wav(
                        engine,
                        seg_text.encode("utf-8"),
                        speaker_embedding,
                        speaker_embedding_len,
                        cfg_ptr,
                        ctypes.byref(out_samples_ptr),
                        ctypes.byref(out_sample_count),
                    )
                    if ret != 0:
                        err = _get_aila_error(lib, engine)
                        raise RuntimeError(f"Aila TTS 合成失败 (code={ret}): {err}")

                    seg_samples = np.ctypeslib.as_array(
                        out_samples_ptr, shape=(out_sample_count.value,)
                    ).copy()
                    lib.aila_free_samples(out_samples_ptr)
                    all_samples.append(seg_samples)

                    # 每段合成后释放 embedding 引用（只在第一段后释放 ref 的）
                    # speaker_embedding 由引擎管理，不需要我们释放

                if len(all_samples) == 1:
                    full_array = all_samples[0]
                else:
                    full_array = np.concatenate(all_samples)

                # 释放 speaker_embedding（如果提取过）
                if speaker_embedding is not None:
                    lib.aila_free_samples(speaker_embedding)

            sample_count = len(full_array)
            print(f"[Aila Synthesizer] Synthesized: {sample_count} samples, "
                  f"{len(segments)} segment(s)")

            # 转为 torch tensor → AUDIO 格式
            audio_tensor = torch.from_numpy(full_array).to(torch.float32)
            audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, N)

            # 显存清理
            cleanup_key = memory_cleanup.split(" (")[0] if " (" in memory_cleanup else memory_cleanup
            if cleanup_key == "full_cleanup":
                print(f"[Aila Synthesizer] 彻底清理: {model_path} + XPU 缓存")
                _shutdown_engine(model_path)
                try:
                    if hasattr(torch, "xpu") and torch.xpu.is_available():
                        torch.xpu.empty_cache()
                        torch.xpu.synchronize()
                except Exception:
                    pass

            return io.NodeOutput({
                "waveform": audio_tensor,
                "sample_rate": TTS_SAMPLE_RATE,
            })

        except Exception as e:
            raise RuntimeError(f"AilaSynthesizer 失败: {e}")
