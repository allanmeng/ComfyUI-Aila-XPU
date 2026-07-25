"""Aila 提示词反推插件 - ComfyUI 插件入口"""

__version__ = "0.1.7.2"

from comfy_api.latest import ComfyExtension, io

from .nodes import AilaModelLoader, AilaCaptioner, AilaASRLoader, AilaTranscriber, AilaTTSLoader, AilaSynthesizer


def _check_bnb():
    """检测 bitsandbytes 状态，仅提示不影响 Aila 运行。"""
    try:
        import torch
    except ImportError:
        return

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return

    try:
        import bitsandbytes as bnb

        try:
            from bitsandbytes.cextension import BNB_BACKEND
            backend = BNB_BACKEND
        except (ImportError, AttributeError):
            backend = "unknown"
        print(f"[Aila] bitsandbytes v{bnb.__version__} (backend={backend})")
    except ImportError:
        print("[Aila] bitsandbytes 未安装（不影响 Aila 引擎推理，仅本地导出 NF4 模型时需安装）")
    except Exception as e:
        print(f"[Aila] bitsandbytes 加载异常: {e}（不影响 Aila 引擎推理）")


_check_bnb()


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
