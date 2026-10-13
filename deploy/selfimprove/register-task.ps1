# Register the nightly GLYDI self-improvement run with Windows Task Scheduler.
#
#     powershell -ExecutionPolicy Bypass -File deploy\selfimprove\register-task.ps1
#     powershell -ExecutionPolicy Bypass -File deploy\selfimprove\register-task.ps1 -Remove
#
# Runs `train\selfimprove.py nightly` at 03:00 as the current user, with the
# repo .env loaded, catching up after a missed run (StartWhenAvailable, the
# Task Scheduler counterpart of Persistent=true), and never on battery.
# Output: train\logs\selfimprove-YYYYMMDD.log and train\logs\task.log.
param([switch]$Remove, [string]$At = "03:00")
$ErrorActionPreference = "Stop"
$Name = "GLYDI self-improvement"
if ($Remove) { Unregister-ScheduledTask -TaskName $Name -Confirm:$false; "removed"; return }

$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$Py = (Get-Command python).Source
# A wrapper that loads .env (KEY=value lines) into the process, then runs nightly.
$Runner = Join-Path $Repo "deploy\selfimprove\run-nightly.ps1"
$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -Python `"$Py`"" -WorkingDirectory $Repo
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DisallowStartIfOnBatteries -StopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $Name -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "train/selfimprove.py nightly (README, Self-improvement)" -Force | Out-Null
Get-ScheduledTask -TaskName $Name | Format-List TaskName, State
"next run: " + (Get-ScheduledTaskInfo -TaskName $Name).NextRunTime
