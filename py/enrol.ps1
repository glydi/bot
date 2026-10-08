# Teach GLYDI a face, on Windows:  py\enrol.ps1 "Kalyan"   (see py\glydi\enrol.py)
$ErrorActionPreference = "Stop"
$HERE = $PSScriptRoot
$venv = "$HERE\.venv\Scripts\python.exe"
if (-not (Test-Path $venv)) { throw "no py\.venv -- run py\setup.ps1 first" }
$env:PYTHONPATH = $HERE
& $venv -m glydi.enrol @args
exit $LASTEXITCODE
