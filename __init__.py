"""Aila 提示词反推插件 - ComfyUI 插件入口"""

__version__ = "0.1.7"

import subprocess
import sys
import os

from comfy_api.latest import ComfyExtension, io

from .nodes import AilaModelLoader, AilaCaptioner, AilaASRLoader, AilaTranscriber, AilaTTSLoader, AilaSynthesizer


def _ensure_bnb_compat():
    """自动检测 Intel Arc 并安装 XPU 版 bitsandbytes"""
    try:
        import torch
    except ImportError:
        return

    # 仅 Intel Arc 显卡需要处理
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return

    # 尝试导入 bitsandbytes
    try:
        import bitsandbytes as bnb

        # 检查是否已正确加载 XPU 后端
        if getattr(bnb, "COMPILED_WITH_CUDA", None):
            raise ImportError("CUDA 版本不兼容 Intel Arc")
        print(f"[Aila] bitsandbytes XPU 版已就绪 (v{bnb.__version__})")
        return
    except ImportError:
        pass
    except Exception:
        pass

    # 自动安装 XPU 版
    print("[Aila] 检测到 Intel Arc 显卡，正在安装 bitsandbytes XPU 版...")
    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install",
                "--force-reinstall", "bitsandbytes",
                "--extra-index-url", "https://pytorch.org/whl/xpu",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[Aila] bitsandbytes XPU 版安装完成，请重启 ComfyUI")
    except Exception as e:
        print(f"[Aila] 自动安装失败，请手动执行：")
        print(f"  {sys.executable} -m pip install --force-reinstall bitsandbytes --extra-index-url https://pytorch.org/whl/xpu")


_ensure_bnb_compat()


class AilaXpuExtension(ComfyExtension):
    """Aila XPU Captioner 扩展。"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AilaModelLoader,
            AilaCaptioner,
            AilaASRLoader,
            AilaTranscriber,
            AilaTTSLoader,
            AilaSynthesizer,
        ]


async def comfy_entrypoint() -> AilaXpuExtension:
    return AilaXpuExtension()
