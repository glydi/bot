# Run the Python build from anywhere, on Windows:
#     py\run.ps1 [--no-camera --no-mic --headless]
#
# `python -m glydi` needs py\ on the module path; this sets it so the
# command works from the repo root, which is where everything else is run
# from.
$ErrorActionPreference = "Stop"
$HERE = $PSScriptRoot
$venv = "$HERE\.venv\Scripts\python.exe"
if (-not (Test-Path $venv)) { throw "no py\.venv -- run py\setup.ps1 first" }
$env:PYTHONPATH = $HERE
& $venv -m glydi @args
exit $LASTEXITCODE
