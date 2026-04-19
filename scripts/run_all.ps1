param(
    [int]$Episodes     = 200,
    [int]$NumAgents    = 2,
    [int]$EvalEpisodes = 50,
    [string]$PlotsDir  = "plots",
    [switch]$SkipTraining
)

$Root = $PSScriptRoot | Split-Path

# ── Phase A: Train all configs ───────────────────────────────────────────────
if ($SkipTraining) {
    Write-Host "=== Phase A: Skipping training ==="
} else {
    Write-Host "=== Phase A: Training ==="
    & "$Root\scripts\train.ps1" -Episodes $Episodes -NumAgents $NumAgents
}

$ModelsRoot = Join-Path $Root "models"
if (-not (Test-Path $ModelsRoot)) {
    Write-Host "No models directory found at '$ModelsRoot'. Skipping evaluate + plot."
    exit 0
}

$AbsPlots = if ([System.IO.Path]::IsPathRooted($PlotsDir)) {
    $PlotsDir
} else {
    Join-Path $Root $PlotsDir
}
New-Item -ItemType Directory -Path $AbsPlots -Force | Out-Null

$EnvDirs = Get-ChildItem -Path $ModelsRoot -Directory

# ── Phase B: Evaluate each environment ───────────────────────────────────────
Write-Host ""
Write-Host "=== Phase B: Evaluating ==="

foreach ($EnvDir in $EnvDirs) {
    $EvalJson = Join-Path $AbsPlots "$($EnvDir.Name)_eval.json"
    $EvalTxt  = Join-Path $AbsPlots "$($EnvDir.Name)_eval.txt"
    Write-Host "Evaluating '$($EnvDir.Name)' -> $EvalJson"
    & "$Root\.venv\Scripts\python.exe" "$Root\scripts\evaluate.py" `
        --models_dir $EnvDir.FullName `
        --n_episodes $EvalEpisodes `
        --save $EvalJson `
        --save_txt $EvalTxt
}

# ── Phase C: Plot results per environment ────────────────────────────────────
Write-Host ""
Write-Host "=== Phase C: Plotting ==="

foreach ($EnvDir in $EnvDirs) {
    $EvalJson = Join-Path $AbsPlots "$($EnvDir.Name)_eval.json"
    $SavePath = Join-Path $AbsPlots "$($EnvDir.Name).png"
    $EvalArg  = if (Test-Path $EvalJson) { @('--eval_results', $EvalJson) } else { @() }
    Write-Host "Plotting '$($EnvDir.Name)' -> $SavePath"
    & "$Root\.venv\Scripts\python.exe" "$Root\scripts\plot.py" `
        --models_dir $EnvDir.FullName `
        --save $SavePath `
        @EvalArg
}

Write-Host ""
Write-Host "Done. Plots and eval results saved to: $AbsPlots"
