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

# Always the latest version: rebuild before starting (a few seconds when
# nothing changed). A failed build is logged and the last good binary
# runs instead. GLYDI_AUTOBUILD=0 skips it.
if ($env:GLYDI_AUTOBUILD -ne "0" -and (Get-Command cargo -ErrorAction SilentlyContinue)) {
    $buildLog = Join-Path $env:TEMP "glydi-build.log"
    Push-Location $PSScriptRoot
    try {
        $ErrorActionPreference = "Continue"
        cargo build --release -p glydi --features vision,kokoro *> $buildLog
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "build failed (see $buildLog); starting the last good version"
        }
    } finally {
        $ErrorActionPreference = "Stop"
        Pop-Location
    }
}

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

# Supervised: a non-zero exit (a panic is 70, a watchdog restart 75, a
# crash anything else) starts the bot again after five seconds, the way
# systemd's Restart=always does on the Jetson. A clean quit (0) or Ctrl-C
# in this window ends the loop. GLYDI_NO_RESTART=1 runs it once.
while ($true) {
    & $exe run @args
    $code = $LASTEXITCODE
    if ($code -eq 0 -or $env:GLYDI_NO_RESTART -eq "1") { exit $code }
    Write-Warning "glydi exited with $code; restarting in 5 s (Ctrl-C to stop)"
    Start-Sleep -Seconds 5
}
