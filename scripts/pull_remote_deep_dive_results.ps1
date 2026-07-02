<#
.SYNOPSIS
Pull lightweight deep-dive outputs required for local analysis and plotting.

.DESCRIPTION
Uses SSH for remote file discovery and SCP for transfer. The strict allowlist
includes results, predictions, figures, logs, deep-dive configs, summaries,
completion markers, and Trainer log history. It excludes checkpoints, model
weights, optimizer state, feature caches, K-means centers, and joblib files.

.EXAMPLE
.\scripts\pull_remote_deep_dive_results.ps1 `
  -HostName 203.0.113.10 `
  -IdentityFile $HOME\.ssh\id_ed25519

.EXAMPLE
.\scripts\pull_remote_deep_dive_results.ps1 `
  -HostName gpu.example.com `
  -UserName root `
  -RemoteProjectRoot /root/workspace/SpeechTech-asr `
  -OverwriteExisting

.EXAMPLE
.\scripts\pull_remote_deep_dive_results.ps1 `
  -HostName gpu.example.com `
  -IdentityFile $HOME\.ssh\id_ed25519 `
  -ListOnly
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HostName,

    [string]$UserName = "root",

    [ValidateRange(1, 65535)]
    [int]$Port = 22,

    [string]$IdentityFile,

    [string]$RemoteProjectRoot = "/root/workspace/SpeechTech-asr",

    [string]$LocalProjectRoot,

    [switch]$OverwriteExisting,

    [switch]$ListOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$syncScript = Join-Path $PSScriptRoot "sync_remote_artifacts.ps1"
if (-not (Test-Path -LiteralPath $syncScript -PathType Leaf)) {
    throw "Required synchronization helper not found: $syncScript"
}

$syncArguments = @{
    HostName = $HostName
    UserName = $UserName
    Port = $Port
    RemoteProjectRoot = $RemoteProjectRoot
    AnalysisOnly = $true
    OverwriteExisting = $OverwriteExisting
    ListOnly = $ListOnly
}
if ($IdentityFile) {
    $syncArguments.IdentityFile = $IdentityFile
}
if ($LocalProjectRoot) {
    $syncArguments.LocalProjectRoot = $LocalProjectRoot
}

Write-Host "Pulling lightweight post-training artifacts only."
Write-Host "Model checkpoints, weights, feature caches, and K-means binaries are excluded."
& $syncScript @syncArguments
