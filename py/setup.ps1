# Set up the Python build's virtual environment, on Windows.
#
#     py\setup.ps1         # create py\.venv and install everything
#
# Python 3.12, not 3.13+: onnxruntime, opencv and insightface wheels lag
# the newest interpreter by months, and a source build of insightface
# needs a compiler and half an hour. Windows PowerShell 5.1 is enough.
$ErrorActionPreference = "Stop"
$HERE = $PSScriptRoot

# $env:PYTHON wins; otherwise the launcher, the default install, then
# whatever `python` is -- the first one that is actually a 3.12.
$candidates = @()
if ($env:PYTHON) { $candidates += ,@($env:PYTHON) }
$candidates += ,@("py", "-3.12")
$candidates += ,@("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe")
$candidates += ,@("python")

$exe = $null
foreach ($c in $candidates) {
    $exe = $c[0]; $extra = @($c | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { $exe = $null; continue }
    try { $v = & $exe @extra -c "import sys; print(sys.version_info[:2] == (3, 12))" 2>$null }
    catch { $v = $null }
    if ("$v" -eq "True") { break }
    $exe = $null
}
if (-not $exe) { throw "no Python 3.12 found; install it from python.org or set `$env:PYTHON" }
Write-Host "python: $(& $exe @extra -V) at $((Get-Command $exe).Source)"

# `venv` is a no-op on an existing .venv, so this is safe to re-run.
& $exe @extra -m venv "$HERE\.venv"
$venv = "$HERE\.venv\Scripts\python.exe"
& $venv -m pip -q install --upgrade pip
& $venv -m pip -q install -r "$HERE\requirements.txt"
@'
import cv2, insightface, numpy, onnxruntime, onnx_asr, requests, sounddevice
print("ok:", numpy.__version__, "onnxruntime", onnxruntime.__version__, "cv2", cv2.__version__)
'@ | & $venv -
if ($LASTEXITCODE -ne 0) { throw "import smoke test failed" }
Write-Host ""
Write-Host "next:  py\run.ps1          (the bot)"
Write-Host "       py\enrol.ps1 NAME   (teach it a face)"
