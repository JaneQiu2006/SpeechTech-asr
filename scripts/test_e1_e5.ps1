param(
    [string[]]$Experiments = @("E1", "E2", "E3", "E4", "E5"),
    [int]$BatchSize = 2,
    [switch]$Overwrite,
    [switch]$NoFp16
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = "D:\miniconda3\envs\ssl_asr\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $Python = "python"
}

$Arguments = @(
    (Join-Path $PSScriptRoot "evaluate_e1_e5.py"),
    "--manifest", (Join-Path $ProjectRoot "data\manifests\test_clean.jsonl"),
    "--data_root", (Join-Path $ProjectRoot "data"),
    "--metrics_path", (Join-Path $ProjectRoot "results\test_metrics.csv"),
    "--predictions_dir", (Join-Path $ProjectRoot "results\predictions"),
    "--batch_size", $BatchSize,
    "--experiments"
) + $Experiments

if ($Overwrite) {
    $Arguments += "--overwrite"
}
if ($NoFp16) {
    $Arguments += "--no_fp16"
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "E1-E5 test evaluation failed with exit code $LASTEXITCODE"
}
