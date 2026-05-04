"""图像处理工具函数"""

import io
import base64

import torch
from PIL import Image


def image_tensor_to_data_uri(
    img_tensor: torch.Tensor,
    max_pixels: int = 4096,
) -> str:
    """
    将 ComfyUI 图像 tensor 转换为 data URI 字符串。
    仅当 image 分辨率超高时做保护性缩放，避免 Aila 视觉编码器 OOM。

    Args:
        img_tensor: Tensor of shape (H, W, C), float [0, 1] range
        max_pixels: 最长边的最大像素值（默认 4096），超过则等比例缩放

    Returns:
        data URI string: "data:image/png;base64,<base64>"
    """
    # 确保是 (H, W, C) 格式，float 在 [0,1] 范围
    if img_tensor.ndim != 3 or img_tensor.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected (H, W, C) tensor, got shape {img_tensor.shape}")

    # 转换到 numpy uint8
    arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")

    # 处理单通道灰度图 -> RGB
    if arr.shape[2] == 1:
        arr = arr.repeat(3, axis=2)

    img = Image.fromarray(arr)

    # 仅当超过 max_pixels 时做保护性缩放
    orig_size = img.size
    if max(orig_size) > max_pixels:
        ratio = max_pixels / max(orig_size)
        new_size = (int(orig_size[0] * ratio), int(orig_size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
