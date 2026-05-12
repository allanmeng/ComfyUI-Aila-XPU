"""Aila 提示词反推插件 - ComfyUI 插件入口"""

__version__ = "0.1.4"

from comfy_api.latest import ComfyExtension, io

from .nodes import AilaModelLoader, AilaCaptioner


class AilaXpuExtension(ComfyExtension):
    """Aila XPU Captioner 扩展。"""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            AilaModelLoader,
            AilaCaptioner,
        ]


async def comfy_entrypoint() -> AilaXpuExtension:
    return AilaXpuExtension()
