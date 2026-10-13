# Called by the scheduled task (register-task.ps1): load the repo .env, run nightly.
param([string]$Python = "python")
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $Repo
$EnvFile = Join-Path $Repo ".env"
if (Test-Path $EnvFile) {
    foreach ($line in Get-Content $EnvFile) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"').Trim("'"), "Process")
        }
    }
}
New-Item -ItemType Directory -Force (Join-Path $Repo "train\logs") | Out-Null
& $Python "train\selfimprove.py" nightly *>> (Join-Path $Repo "train\logs\task.log")
exit $LASTEXITCODE
