param(
    [int]$Episodes = 10000,
    [string]$Config = "configs/ppo_note.yaml",
    [string]$SavePrefix = "models/ppo_note",
    [int]$NumAgents = 5
)

$Root = $PSScriptRoot | Split-Path

$Jobs = 1..$NumAgents | ForEach-Object {
    $Id = $_
    $Save = "${SavePrefix}_run${Id}.pth"
    Start-Job -ScriptBlock {
        param($Root, $Config, $Save, $Episodes)
        Set-Location $Root
        & "$Root\.venv\Scripts\python.exe" "$Root\scripts\train.py" `
            --config $Config `
            --save $Save `
            --n_episodes $Episodes `
            --no_plot
    } -ArgumentList $Root, $Config, $Save, $Episodes
}

Write-Host "Started $NumAgents training jobs. Waiting for completion..."
$Jobs | Wait-Job | Receive-Job
$Jobs | Remove-Job
Write-Host "All training jobs complete."
