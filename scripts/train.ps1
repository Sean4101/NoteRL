param(
    [int]$Episodes = 5000,
    [int]$NumAgents = 3
)

$Root = $PSScriptRoot | Split-Path

$Configs = Get-ChildItem -Path "$Root\configs" -Filter "*.yaml" -Recurse

foreach ($ConfigFile in $Configs) {
    $RelConfig = $ConfigFile.FullName.Substring($Root.Length + 1) -replace '\\', '/'
    $EnvName   = $ConfigFile.Directory.Name
    $ConfigName = $ConfigFile.BaseName
    $SaveDir   = "$Root\models\$EnvName"
    New-Item -ItemType Directory -Path $SaveDir -Force | Out-Null

    Write-Host "Training config: $RelConfig"

    $Jobs = 1..$NumAgents | ForEach-Object {
        $Id = $_
        $Save = "$SaveDir\${ConfigName}_run${Id}.pth"
        Start-Job -ScriptBlock {
            param($Root, $Config, $Save, $Episodes)
            Set-Location $Root
            & "$Root\.venv\Scripts\python.exe" "$Root\scripts\train.py" `
                --config $Config `
                --save $Save `
                --n_episodes $Episodes `
                --no_plot
        } -ArgumentList $Root, $RelConfig, $Save, $Episodes
    }

    Write-Host "  Started $NumAgents jobs. Waiting for completion..."
    $Jobs | Wait-Job | Receive-Job
    $Jobs | Remove-Job
    Write-Host "  Done: $RelConfig"
}

Write-Host "All configs trained."
