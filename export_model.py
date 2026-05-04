#!python
"""
Aila 模型导出工具

将 Hugging Face 模型量化为 bitsandbytes NF4 格式，导出为 Aila 引擎可加载的目录结构。

用法:
  # 从 Hugging Face 下载并导出
  python export_model.py --from-hf Qwen/Qwen3.5-0.8B

  # 从本地路径导出
  python export_model.py --source-model D:/models/Qwen3.5-0.8B

  # 指定输出目录
  python export_model.py --from-hf Qwen/Qwen3.5-0.8B --export-path ./models/aila/qwen3.5-0.8B-bnb-nf4

  # 保留视觉编码器为密集精度（多模态模型默认自动保留）
  python export_model.py --from-hf Qwen/Qwen3.5-4B

依赖:
  - torch (with XPU support)
  - bitsandbytes
  - transformers >= 4.46.0
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ─── 默认路径 ───────────────────────────────────────────────────────────────

HERE = Path(__file__).parent.resolve()
DEFAULT_EXPORT_DIR = HERE.parent.parent / "models" / "aila"  # models/aila/


# ─── 辅助函数 ───────────────────────────────────────────────────────────────

def eprint(*args, **kwargs):
    """打印到 stderr（不影响 stdout 可能的管道输出）。"""
    print(*args, file=sys.stderr, **kwargs, flush=True)


def configure_stdout() -> None:
    """确保 stdout 支持 UTF-8。"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def is_multimodal_config(config: AutoConfig) -> bool:
    """检查模型配置是否为多模态（含视觉）。"""
    return (
        getattr(config, "model_type", None) == "qwen3_5"
        and getattr(config, "vision_config", None) is not None
    )


def resolve_keep_dense_modules(
    config: AutoConfig,
    requested_modules: Optional[list[str]],
) -> list[str]:
    """确定哪些模块保持密集精度（不量化）。

    多模态模型默认保留 model.visual（视觉编码器）不量化。
    """
    modules = list(requested_modules or [])
    if is_multimodal_config(config):
        modules.append("model.visual")
    return list(dict.fromkeys(modules))  # 去重保持顺序


# ─── 主流程 ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hugging Face 模型 → Aila NF4 量化格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python export_model.py --from-hf Qwen/Qwen3.5-0.8B
  python export_model.py --source-model /path/to/Qwen3.5-0.8B --export-path ./models/aila/my-model
        """,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-hf", type=str, default=None,
        help="Hugging Face 模型 ID（如 Qwen/Qwen3.5-0.8B），自动下载",
    )
    src.add_argument(
        "--source-model", type=Path, default=None,
        help="本地模型路径",
    )

    parser.add_argument(
        "--export-path", type=Path, default=None,
        help=f"输出目录（默认: {DEFAULT_EXPORT_DIR}/<model-name>-bnb-nf4-offline）",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser.add_argument(
        "--keep-dense-module", action="append", dest="keep_dense_modules",
        default=None,
        help="保持密集精度的模块前缀（可重复使用），多模态视觉编码器默认保留",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="不使用 Hugging Face 缓存，强制重新下载",
    )

    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    """准备并验证输出目录。"""
    if output_dir.exists():
        if overwrite:
            eprint(f"  [清空] {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}\n"
                "  使用 --overwrite 来覆盖重建。"
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_processor_assets(source_dir: Path, output_dir: Path) -> list[str]:
    """复制多模态模型的 processor 配置（如图像处理器配置）。"""
    copied: list[str] = []
    for path in sorted(source_dir.glob("*processor_config.json")):
        shutil.copy2(path, output_dir / path.name)
        copied.append(path.name)
    return copied


def check_environment() -> None:
    """环境检查：torch.xpu + bitsandbytes。"""
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError(
            "torch.xpu 不可用。请确保已安装 Intel XPU 版 PyTorch。\n"
            "  pip install torch --index-url https://pytorch.org/whl/xpu"
        )
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise RuntimeError(
            "bitsandbytes 未安装。\n"
            "  pip install bitsandbytes"
        )
    except Exception as e:
        raise RuntimeError(
            f"bitsandbytes 加载失败: {e}\n"
            "  请检查 bitsandbytes 是否与你的 PyTorch 版本兼容。"
        )


def main() -> int:
    configure_stdout()
    args = parse_args()

    # ── 环境检查 ──
    eprint("=" * 60)
    eprint("Aila 模型导出工具")
    eprint("=" * 60)
    check_environment()

    import bitsandbytes as bnb
    import bitsandbytes.cextension as bnb_cext

    # ── 确定源路径 ──
    if args.from_hf:
        source_path_str = args.from_hf
        eprint(f"\n📥 源: Hugging Face [bold]{source_path_str}[/]")
        # 不需要检查本地存在
    else:
        source_path = args.source_model.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"本地模型路径不存在: {source_path}")
        source_path_str = str(source_path)
        eprint(f"\n📂 源: 本地 [bold]{source_path}[/]")

    # ── 确定输出路径 ──
    # 从源路径提取模型名称
    if args.from_hf:
        model_name = args.from_hf.rstrip("/").split("/")[-1]
    else:
        model_name = args.source_model.name

    export_path = args.export_path or (
        DEFAULT_EXPORT_DIR / f"{model_name.lower()}-bnb-nf4-offline"
    )
    export_path = export_path.resolve()

    # ── 硬件信息 ──
    device = "xpu:0"
    eprint(f"\n🔧 环境:")
    eprint(f"   torch:     {torch.__version__}")
    eprint(f"   bitsandbytes: {bnb.__version__} (backend={bnb_cext.BNB_BACKEND})")
    eprint(f"   显卡:      {torch.xpu.get_device_name(0)}")
    eprint(f"   设备:      {device}")
    eprint(f"   量化:      NF4, compute_dtype=float16, double_quant=True")

    # ── 加载配置 ──
    eprint(f"\n📋 模型配置:")
    eprint(f"   源:         {source_path_str}")
    eprint(f"   输出:       {export_path}")

    torch.manual_seed(args.seed)

    # 从 Hugging Face 加载时，需要传入 revision/trust info
    hf_kwargs = {}
    if args.from_hf:
        hf_kwargs["revision"] = "main"
    if args.no_cache:
        hf_kwargs["force_download"] = True

    config = AutoConfig.from_pretrained(
        source_path_str, trust_remote_code=True, **hf_kwargs
    )
    multimodal = is_multimodal_config(config)
    keep_dense_modules = resolve_keep_dense_modules(config, args.keep_dense_modules)

    model_loader = AutoModelForImageTextToText if multimodal else AutoModelForCausalLM

    eprint(f"   架构:       {getattr(config, 'model_type', 'unknown')}")
    eprint(f"   多模态:     {'是' if multimodal else '否'}")
    eprint(f"   参数:       {model_loader.__name__}")
    eprint(f"   密集模块:   {keep_dense_modules or '无'}")

    # ── 准备输出目录 ──
    prepare_output_dir(export_path, args.overwrite)

    # ── 量化配置 ──
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=keep_dense_modules or None,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 加载分词器 ──
    eprint(f"\n⏳ 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        source_path_str, trust_remote_code=True, **hf_kwargs
    )

    # ── 加载并量化模型 ──
    eprint(f"⏳ 加载并量化模型（NF4）...")
    load_start = time.perf_counter()

    model = model_loader.from_pretrained(
        source_path_str,
        quantization_config=quant_config,
        device_map={"": device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **hf_kwargs,
    )
    model.eval()
    load_seconds = time.perf_counter() - load_start
    eprint(f"   加载完成: {load_seconds:.1f} 秒")

    # ── 保存 ──
    eprint(f"\n💾 保存到: {export_path}")
    save_start = time.perf_counter()

    model.save_pretrained(export_path, safe_serialization=True)
    tokenizer.save_pretrained(export_path)

    processor_assets = copy_processor_assets(
        Path(source_path) if not args.from_hf else export_path.parent / model_name,
        export_path,
    )

    # 对于 Hugging Face 模型，processor 文件可能在缓存中
    if args.from_hf and not processor_assets:
        from transformers.utils import cached_file
        for suffix in ["preprocessor_config.json", "processor_config.json"]:
            try:
                cached = cached_file(source_path_str, suffix, **hf_kwargs)
                if cached:
                    shutil.copy2(cached, export_path / suffix)
                    processor_assets.append(suffix)
            except Exception:
                pass

    save_seconds = time.perf_counter() - save_start
    eprint(f"   保存完成: {save_seconds:.1f} 秒")

    # ── 验证 ──
    saved_files = sorted(p.name for p in export_path.iterdir())
    saved_config = json.loads((export_path / "config.json").read_text(encoding="utf-8"))

    eprint(f"\n📦 导出结果:")
    eprint(f"   文件数:     {len(saved_files)}")
    total_size = sum(
        f.stat().st_size for f in export_path.iterdir() if f.is_file()
    )
    eprint(f"   总大小:     {total_size / 1024 / 1024:.0f} MB")
    eprint(f"   模型类型:   {saved_config.get('model_type', 'unknown')}")
    eprint(f"   已量化:     {'是' if 'quantization_config' in saved_config else '否'}")
    eprint(f"   视觉配置:   {'有' if 'vision_config' in saved_config else '无'}")
    if processor_assets:
        eprint(f"   processor:  {processor_assets}")

    eprint(f"\n✅ 导出成功！")
    eprint(f"   输出目录: {export_path}")
    eprint()
    eprint(f"   在 ComfyUI 中:")
    eprint(f"     1. 重启 ComfyUI")
    eprint(f"     2. 添加节点 Aila > Aila Model Loader")
    eprint(f"     3. 选择模型: {export_path.name}")
    eprint(f"     4. 连接 Aila Captioner 节点")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
