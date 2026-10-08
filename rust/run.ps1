# Run the Rust build on Windows: hand every argument to `glydi run`.
#
#   rust\run.ps1 --headless --no-camera --no-mic --text
#
# The binary loads the repository .env itself (rust/glydi/src/main.rs,
# `dotenv::apply`): KEY=value lines, `export` tolerated, one layer of
# quotes stripped, and a variable already set in this shell is never
# replaced -- so `$env:GLYDI_TTS = "mac"; rust\run.ps1` still wins over
# the file. This script used to apply the file itself before the binary
# could; it no longer needs to.
# PowerShell 5.1 is enough.

$ErrorActionPreference = "Stop"

$release = Join-Path $PSScriptRoot "target\release\glydi.exe"
$debug = Join-Path $PSScriptRoot "target\debug\glydi.exe"
if (Test-Path -LiteralPath $release) {
    $exe = $release
} elseif (Test-Path -LiteralPath $debug) {
    $exe = $debug
} else {
    Write-Error "no glydi.exe under $PSScriptRoot\target: build it first with`n  cargo build --release -p glydi --features vision,kokoro"
    exit 1
}

& $exe run @args
exit $LASTEXITCODE
