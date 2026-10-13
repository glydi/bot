# Start the Whisper STT service for GLYDI_STT=remote (rust/sense-audio/src/remote.rs).
#   services\stt\run.ps1                       # large-v3-turbo on 127.0.0.1:8178
#   services\stt\run.ps1 -Model distil-large-v3
# Needs train\.venv-cuda (PyTorch cu128) with faster-whisper installed:
#   train\.venv-cuda\Scripts\python.exe -m pip install faster-whisper
# First start downloads the model (~1.6 GB) into the Hugging Face cache.
param(
    [string]$Model = "large-v3-turbo",
    [int]$Port = 8178,
    [string]$Prompt = ""
)
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$py = Join-Path $root "train\.venv-cuda\Scripts\python.exe"
if (-not (Test-Path $py)) { Write-Error "missing $py (see train/README.md)"; exit 1 }
# Windows without Developer Mode cannot symlink in the HF cache; copy instead.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$extra = @()
if ($Prompt) { $extra = @("--prompt", $Prompt) }
& $py (Join-Path $PSScriptRoot "server.py") --model $Model --port $Port @extra
