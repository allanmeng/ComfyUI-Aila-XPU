<#
.SYNOPSIS
    Aila Captioner Plugin - 一键安装运行时和配置
.DESCRIPTION
    从 GitHub 下载 Aila 发行包，解压 AilaShared.dll 及运行时依赖，
    自动生成 config.json，完成插件所需的运行环境配置。
.NOTES
    作者: 小深 ⚡
    版本: 0.1.0
    需要: Windows 10+ / PowerShell 5.1+
#>

$ErrorActionPreference = "Stop"

# ─── 配置 ──────────────────────────────────────────────────────────────────
$AILA_VERSION      = "0.1.0"
$RELEASE_URL       = "https://github.com/Blackwood416/Aila/releases/download/$AILA_VERSION/Aila-v$AILA_VERSION-win64.zip"
$ZIP_FILENAME      = "Aila-v$AILA_VERSION-win64.zip"
$ZIP_EXPECTED_SIZE = 157MB   # 约 157 MB

# 脚本所在目录即插件根目录
$PLUGIN_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$RUNTIME_DIR = Join-Path $PLUGIN_DIR "aila_runtime"
$CONFIG_FILE = Join-Path $PLUGIN_DIR "config.json"

# 模型目录（相对于 ComfyUI 根）
$COMFY_DIR = Resolve-Path (Join-Path $PLUGIN_DIR "..\..\..\")  # custom_nodes/ComfyUI-Aila-XPU -> ComfyUI
$MODEL_DIR = Join-Path $COMFY_DIR "models" "aila"

# ─── 辅助函数 ──────────────────────────────────────────────────────────────

function Write-Banner {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "║      Aila Captioner - 运行时安装助手     ║" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Message)
    Write-Host ">> $Message" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "   $Message" -ForegroundColor Gray
}

# ─── 主流程 ────────────────────────────────────────────────────────────────

try {
    Write-Banner

    # ── 步骤 1: 检测现有安装 ──
    Write-Step "步骤 1/4: 检测现有安装"
    $existingDll = Get-ChildItem -Path $RUNTIME_DIR -Recurse -Filter "AilaShared.dll" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($existingDll) {
        Write-Success "检测到已有运行时: $($existingDll.FullName)"
        $reinstall = Read-Host "  是否重新下载安装？(y/N)"
        if ($reinstall -ne "y" -and $reinstall -ne "Y") {
            Write-Host "  跳过下载，保留现有运行时。" -ForegroundColor Gray
            $SKIP_DOWNLOAD = $true
        } else {
            $SKIP_DOWNLOAD = $false
            Write-Host "  将重新下载安装..." -ForegroundColor Gray
        }
    } else {
        $SKIP_DOWNLOAD = $false
        Write-Info "未检测到现有运行时，将进行全新安装。"
    }

    # ── 步骤 2: 下载 Aila 发行包 ──
    if (-not $SKIP_DOWNLOAD) {
        Write-Step "步骤 2/4: 下载 Aila v$AILA_VERSION 发行包 (~157 MB)"

        $zipPath = Join-Path $env:TEMP $ZIP_FILENAME

        # 检查是否已有缓存的 zip
        if (Test-Path $zipPath -PathType Leaf) {
            $size = (Get-Item $zipPath).Length
            if ($size -ge $ZIP_EXPECTED_SIZE * 0.8) {
                Write-Info "使用缓存文件: $zipPath (已存在)"
            } else {
                Write-Info "缓存文件不完整，重新下载..."
                Remove-Item $zipPath -Force
                Invoke-WebRequest -Uri $RELEASE_URL -OutFile $zipPath -UseBasicParsing -Verbose
            }
        } else {
            Write-Info "正在下载... (可能会持续几分钟，视网络而定)"
            Invoke-WebRequest -Uri $RELEASE_URL -OutFile $zipPath -UseBasicParsing
        }

        # 验证文件
        if (-not (Test-Path $zipPath)) {
            throw "下载失败: $zipPath 不存在"
        }
        Write-Success "下载完成: $((Get-Item $zipPath).Length / 1MB -as [int]) MB"

        # ── 步骤 3: 解压 ──
        Write-Step "步骤 3/4: 解压运行时文件"

        # 清空现有运行时目录
        if (Test-Path $RUNTIME_DIR) {
            Remove-Item -Path "$RUNTIME_DIR\*" -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            New-Item -ItemType Directory -Path $RUNTIME_DIR -Force | Out-Null
        }

        Write-Info "解压到: $RUNTIME_DIR"
        Expand-Archive -Path $zipPath -DestinationPath $RUNTIME_DIR -Force

        # 找到 AilaShared.dll
        $dllFile = Get-ChildItem -Path $RUNTIME_DIR -Recurse -Filter "AilaShared.dll" | Select-Object -First 1
        if (-not $dllFile) {
            throw "解压后未找到 AilaShared.dll，请检查发行包结构"
        }
        $DLL_PATH = $dllFile.FullName
        Write-Success "运行时已安装: $DLL_PATH"

        # 显示解压内容
        $fileCount = (Get-ChildItem -Path $RUNTIME_DIR -Recurse -File).Count
        $totalSize = (Get-ChildItem -Path $RUNTIME_DIR -Recurse -File | Measure-Object Length -Sum).Sum
        Write-Info "共解压 $fileCount 个文件 ($($totalSize / 1MB -as [int]) MB)"

        # 清理 zip
        Remove-Item $zipPath -Force
        Write-Info "临时文件已清理"

        # 更新 DLL_PATH 变量（可能是在子目录中）
        $dllFile = Get-ChildItem -Path $RUNTIME_DIR -Recurse -Filter "AilaShared.dll" | Select-Object -First 1
        $DLL_PATH = $dllFile.FullName
        Write-Success "运行时已安装: $DLL_PATH"
    } else {
        # 已有安装，获取 DLL 路径
        $dllFile = Get-ChildItem -Path $RUNTIME_DIR -Recurse -Filter "AilaShared.dll" | Select-Object -First 1
        $DLL_PATH = $dllFile.FullName
    }

    # ── 确保模型目录存在 ──
    if (-not (Test-Path $MODEL_DIR)) {
        New-Item -ItemType Directory -Path $MODEL_DIR -Force | Out-Null
        Write-Info "已创建模型目录: $MODEL_DIR"
    }

    # ── 步骤 4: 生成 config.json ──
    Write-Step "步骤 4/4: 生成配置文件"

    # 检查现有 config，保留用户自定义的 model_folders
    $existingModelFolders = @()
    if (Test-Path $CONFIG_FILE -PathType Leaf) {
        try {
            $existingConfig = Get-Content $CONFIG_FILE -Raw | ConvertFrom-Json
            if ($existingConfig.model_folders -and $existingConfig.model_folders.Count -gt 0) {
                $existingModelFolders = @($existingConfig.model_folders)
                Write-Info "保留已有的 model_folders 设置"
            }
        } catch {
            Write-Info "现有 config.json 格式有误，将覆盖"
        }
    }

    $config = @{
        dll_path      = $DLL_PATH -replace '\\', '/'   # 统一用正斜杠，Python 兼容
        model_folders = $existingModelFolders + @($MODEL_DIR -replace '\\', '/') | Select-Object -Unique
    }

    $config | ConvertTo-Json -Depth 3 | Set-Content $CONFIG_FILE -Encoding UTF8
    Write-Success "配置文件已生成: $CONFIG_FILE"
    Write-Info "DLL 路径: $($config.dll_path)"
    Write-Info "模型目录: $($config.model_folders -join ', ')"

    # ── 完成 ──
    Write-Host ""
    Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan
    Write-Success "Aila Captioner 运行时安装完成！"
    Write-Host ""
    Write-Host "  接下来请将 Hugging Face 模型导出为 Aila 格式:" -ForegroundColor White
    Write-Host "    python export_model.py --source-model Qwen/Qwen3.5-0.8B" -ForegroundColor Green
    Write-Host ""
    Write-Host "  然后重启 ComfyUI，添加节点:" -ForegroundColor White
    Write-Host "    Aila > Aila Model Loader → Aila Captioner" -ForegroundColor Green
    Write-Host ""
    Write-Host "══════════════════════════════════════════" -ForegroundColor Cyan

} catch {
    Write-Error "安装失败: $_"
    Write-Host "  详情: $($_.ScriptStackTrace)" -ForegroundColor DarkGray
    exit 1
}
