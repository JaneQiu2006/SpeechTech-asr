param(
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

Push-Location $ProjectRoot
try {
    $StandardIds = @("E6A", "E6B", "E8", "E9", "E12A")
    $AvailableIds = @()
    foreach ($Id in $StandardIds) {
        $Names = @{
            "E6A" = "wav2vec2_top6_finetune_10h"
            "E6B" = "wav2vec2_top3_finetune_10h"
            "E8" = "wav2vec2_masking_finetune_10h"
            "E9" = "wav2vec2_frozen_bilstm_10h"
            "E12A" = "wav2vec2_finetune_1h_time_mask"
        }
        if (Test-Path -LiteralPath (Join-Path "exp" $Names[$Id]) -PathType Container) {
            $AvailableIds += $Id
        }
    }
    if ($AvailableIds.Count -gt 0) {
        $Arguments = @(
            "scripts/evaluate_e1_e5.py",
            "--experiments"
        ) + $AvailableIds + @("--batch_size", $BatchSize)
        if ($Overwrite) {
            $Arguments += "--overwrite"
        }
        if ($NoFp16) {
            $Arguments += "--no_fp16"
        }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Standard extension evaluation failed with exit code $LASTEXITCODE"
        }
    }

    $RepresentationExperiments = @()
    foreach ($Layer in @(6, 9, 12)) {
        $Name = "wav2vec2_layer${Layer}_bilstm_ctc"
        if (Test-Path -LiteralPath "exp\$Name\completion.json" -PathType Leaf) {
            $RepresentationExperiments += @{
                Name = $Name
                Model = "exp\$Name"
                Centers = $null
            }
        }
    }
    $BestLayerPath = "artifacts\kmeans\best_layer.txt"
    if (Test-Path -LiteralPath $BestLayerPath -PathType Leaf) {
        $BestLayer = (Get-Content -LiteralPath $BestLayerPath -Raw).Trim()
        foreach ($K in @(50, 100, 200)) {
            $Name = "wav2vec2_discrete_k${K}_bilstm_ctc"
            $Centers = "artifacts\kmeans\wav2vec2_layer${BestLayer}_k${K}\centers.npy"
            if (
                (Test-Path -LiteralPath "exp\$Name\completion.json" -PathType Leaf) -and
                (Test-Path -LiteralPath $Centers -PathType Leaf)
            ) {
                $RepresentationExperiments += @{
                    Name = $Name
                    Model = "exp\$Name"
                    Centers = $Centers
                }
            }
        }
    }

    foreach ($Experiment in $RepresentationExperiments) {
        $Arguments = @(
            "scripts/evaluate_representation_ctc.py",
            "--experiment", $Experiment.Name,
            "--model_dir", $Experiment.Model,
            "--prediction_path", "results\predictions\$($Experiment.Name)_test.jsonl",
            "--batch_size", $BatchSize
        )
        if ($null -ne $Experiment.Centers) {
            $Arguments += @("--centers", $Experiment.Centers)
        }
        if ($Overwrite) {
            $Arguments += "--overwrite"
        }
        if ($NoFp16) {
            $Arguments += "--no_fp16"
        }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$($Experiment.Name) evaluation failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
