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

    [switch]$IncludeTrainingState,

    [switch]$AnalysisOnly,

    [switch]$OverwriteExisting,

    [switch]$ListOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $LocalProjectRoot) {
    $scriptDirectory = $PSScriptRoot
    if (-not $scriptDirectory) {
        $scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    $LocalProjectRoot = Split-Path -Parent $scriptDirectory
}

function Assert-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install/enable Windows OpenSSH Client first."
    }
}

Assert-CommandExists "ssh"
Assert-CommandExists "scp"

if ($RemoteProjectRoot.Contains("'")) {
    throw "RemoteProjectRoot cannot contain a single quote."
}

$localRoot = [System.IO.Path]::GetFullPath($LocalProjectRoot)
if (-not (Test-Path -LiteralPath $localRoot -PathType Container)) {
    throw "Local project root does not exist: $localRoot"
}

$target = "$UserName@$HostName"
$temporaryKeyDirectory = $null
$temporaryPrivateKey = $null
$temporaryAuthorizedLine = $null
$temporaryKeyInstalled = $false

# Enumerate all regular files once. Missing optional directories are ignored.
$remoteListCommand = @'
cd '__REMOTE_ROOT__' || exit 1
for directory in exp results logs artifacts/kmeans artifacts/deep_dive configs/deep_dive; do
    if [ -d "$directory" ]; then
        find "$directory" -type f -print
    fi
done
'@
$remoteListCommand = $remoteListCommand.Replace(
    "__REMOTE_ROOT__",
    $RemoteProjectRoot
)

try {
    $identityArgs = @()
    if ($IdentityFile) {
        $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
        $identityArgs = @(
            "-i", $resolvedIdentity,
            "-o", "IdentitiesOnly=yes"
        )
    }
    else {
        Assert-CommandExists "ssh-keygen"

        # Windows OpenSSH does not reliably support ControlMaster. Bootstrap an
        # ephemeral key with one password-authenticated SSH call, use that key
        # for every SCP call, and remove it from authorized_keys in finally.
        $temporaryKeyDirectory = Join-Path (
            [System.IO.Path]::GetTempPath()
        ) "speechtech-sync-$PID-$([guid]::NewGuid().ToString('N'))"
        [void](New-Item -ItemType Directory -Path $temporaryKeyDirectory)
        $temporaryPrivateKey = Join-Path $temporaryKeyDirectory "id_ed25519"
        $keyComment = "speechtech-sync-$PID-$([guid]::NewGuid().ToString('N'))"

        & ssh-keygen -q -t ed25519 -N '""' -C $keyComment -f $temporaryPrivateKey
        if ($LASTEXITCODE -ne 0) {
            throw "Could not generate the temporary SSH key (exit code $LASTEXITCODE)."
        }

        $publicKey = (Get-Content -LiteralPath "$temporaryPrivateKey.pub" -Raw).Trim()
        if (-not $publicKey -or $publicKey.Contains("'")) {
            throw "Generated an invalid temporary public key."
        }
        $temporaryAuthorizedLine = (
            "no-agent-forwarding,no-port-forwarding,no-X11-forwarding,no-pty " +
            $publicKey
        )

        $bootstrapCommand = @'
umask 077
mkdir -p "$HOME/.ssh"
touch "$HOME/.ssh/authorized_keys"
if ! grep -Fqx '__AUTHORIZED_LINE__' "$HOME/.ssh/authorized_keys"; then
    printf '%s\n' '__AUTHORIZED_LINE__' >> "$HOME/.ssh/authorized_keys"
fi
'@
        $bootstrapCommand = $bootstrapCommand.Replace(
            "__AUTHORIZED_LINE__",
            $temporaryAuthorizedLine
        )

        Write-Host "Installing a temporary transfer key on $target ..."
        Write-Host "Enter the remote account password once when prompted."
        $bootstrapArgs = @(
            "-p", $Port.ToString(),
            "-o", "PubkeyAuthentication=no",
            "-o", "PreferredAuthentications=keyboard-interactive,password"
        )
        & ssh @bootstrapArgs $target $bootstrapCommand
        if ($LASTEXITCODE -ne 0) {
            throw "Could not install the temporary transfer key (exit code $LASTEXITCODE)."
        }
        $temporaryKeyInstalled = $true
        $identityArgs = @(
            "-i", $temporaryPrivateKey,
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes"
        )
    }

    $sshArgs = @("-p", $Port.ToString()) + $identityArgs
    $scpArgs = @("-P", $Port.ToString()) + $identityArgs

    Write-Host "Listing remote artifacts from ${target}:$RemoteProjectRoot ..."
    $remoteFiles = @(& ssh @sshArgs $target $remoteListCommand)
    if ($LASTEXITCODE -ne 0) {
        throw "Remote file listing failed with exit code $LASTEXITCODE."
    }

    $requiredExpNames = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    @(
        "model.safetensors",
        "model.pt",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "trainer_state.json",
        "training_args.bin",
        "completion.json",
        "summary.json"
    ) | ForEach-Object {
        [void]$requiredExpNames.Add($_)
    }

    $selectedFiles = foreach ($line in $remoteFiles) {
        $relativePath = ([string]$line).Trim().Replace("\", "/")
        if (-not $relativePath) {
            continue
        }

        if (
            $relativePath.StartsWith("/") -or
            $relativePath -eq ".." -or
            $relativePath.StartsWith("../") -or
            $relativePath.Contains("/../")
        ) {
            throw "Unsafe relative path returned by remote host: $relativePath"
        }

        if ($AnalysisOnly) {
            # Strict allowlist for post-training analysis. In particular this
            # excludes model.pt, *.safetensors, *.bin, checkpoints, feature
            # caches, K-means centers and serialized K-means models.
            $analysisName = [System.IO.Path]::GetFileName($relativePath)
            $analysisExtension = [System.IO.Path]::GetExtension(
                $relativePath
            ).ToLowerInvariant()
            if (
                $relativePath.Contains("/checkpoint-") -or
                $analysisExtension -in @(
                    ".pt", ".pth", ".bin", ".safetensors", ".joblib",
                    ".npy", ".npz", ".ckpt"
                )
            ) {
                continue
            }

            if (
                $relativePath.StartsWith("results/") -or
                $relativePath.StartsWith("logs/") -or
                $relativePath.StartsWith("configs/deep_dive/")
            ) {
                $relativePath
                continue
            }

            if ($relativePath.StartsWith("exp/deep_dive/")) {
                if (
                    $analysisName -in @(
                        "summary.json",
                        "completion.json",
                        "config.json",
                        "trainer_state.json"
                    )
                ) {
                    $relativePath
                }
                continue
            }

            if ($relativePath.StartsWith("artifacts/deep_dive/")) {
                if (
                    $analysisExtension -in @(
                        ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml", ".md"
                    )
                ) {
                    $relativePath
                }
                continue
            }

            if ($relativePath.StartsWith("artifacts/kmeans/")) {
                if ($analysisName -in @("summary.json", "best_layer.txt")) {
                    $relativePath
                }
                continue
            }

            continue
        }

        if ($relativePath.StartsWith("results/") -or $relativePath.StartsWith("logs/")) {
            $relativePath
            continue
        }

        if ($relativePath.StartsWith("artifacts/kmeans/")) {
            $artifactName = [System.IO.Path]::GetFileName($relativePath)
            if (
                $artifactName -in @(
                    "best_layer.txt",
                    "centers.npy",
                    "kmeans.joblib",
                    "summary.json"
                )
            ) {
                $relativePath
            }
            continue
        }

        if ($relativePath.StartsWith("exp/")) {
            if ($IncludeTrainingState) {
                $relativePath
                continue
            }

            $fileName = [System.IO.Path]::GetFileName($relativePath)
            if ($requiredExpNames.Contains($fileName)) {
                $relativePath
            }
        }
    }

    $selectedFiles = @($selectedFiles | Sort-Object -Unique)
    if ($selectedFiles.Count -eq 0) {
        Write-Host "No matching remote artifacts were found."
        exit 0
    }

    if ($ListOnly) {
        Write-Host "Selected remote files (no download requested):"
        $selectedFiles | ForEach-Object { Write-Host "  $_" }
        Write-Host "Total: $($selectedFiles.Count)"
        exit 0
    }

    $downloaded = 0
    $skipped = 0
    $failed = 0

    foreach ($relativePath in $selectedFiles) {
        $localPath = Join-Path $localRoot ($relativePath.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
        if (
            (Test-Path -LiteralPath $localPath -PathType Leaf) -and
            -not $OverwriteExisting
        ) {
            Write-Host "[skip] $relativePath"
            $skipped++
            continue
        }

        $localDirectory = Split-Path -Parent $localPath
        if (-not (Test-Path -LiteralPath $localDirectory -PathType Container)) {
            [void](New-Item -ItemType Directory -Path $localDirectory -Force)
        }

        $remotePath = "$RemoteProjectRoot/$relativePath"
        $remoteSpec = "${target}:$remotePath"
        Write-Host "[get ] $relativePath"
        $temporaryLocalPath = "$localPath.download-$PID"
        & scp @scpArgs $remoteSpec $temporaryLocalPath
        if ($LASTEXITCODE -eq 0) {
            Move-Item -LiteralPath $temporaryLocalPath -Destination $localPath -Force
            $downloaded++
        }
        else {
            Write-Warning "Failed to download $relativePath (scp exit code $LASTEXITCODE)."
            $failed++
            if (Test-Path -LiteralPath $temporaryLocalPath -PathType Leaf) {
                Remove-Item -LiteralPath $temporaryLocalPath -Force
            }
        }
    }

    Write-Host ""
    Write-Host "Synchronization complete."
    Write-Host "  selected:   $($selectedFiles.Count)"
    Write-Host "  downloaded: $downloaded"
    Write-Host "  skipped:    $skipped"
    Write-Host "  failed:     $failed"
    Write-Host "  local root: $localRoot"

    if ($failed -gt 0) {
        exit 1
    }
}
finally {
    if ($temporaryKeyInstalled) {
        Write-Host "Removing the temporary transfer key from $target ..."
        $cleanupCommand = @'
temporary_file="$(mktemp)"
grep -Fvx '__AUTHORIZED_LINE__' "$HOME/.ssh/authorized_keys" > "$temporary_file" || true
cat "$temporary_file" > "$HOME/.ssh/authorized_keys"
rm -f "$temporary_file"
'@
        $cleanupCommand = $cleanupCommand.Replace(
            "__AUTHORIZED_LINE__",
            $temporaryAuthorizedLine
        )
        $cleanupArgs = @(
            "-p", $Port.ToString(),
            "-i", $temporaryPrivateKey,
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes"
        )
        & ssh @cleanupArgs $target $cleanupCommand
        if ($LASTEXITCODE -ne 0) {
            Write-Warning (
                "Could not remove temporary remote key '$temporaryAuthorizedLine'. " +
                "Remove that exact line from ~/.ssh/authorized_keys manually."
            )
        }
    }

    if ($temporaryPrivateKey) {
        Remove-Item -LiteralPath $temporaryPrivateKey -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath "$temporaryPrivateKey.pub" -Force -ErrorAction SilentlyContinue
    }
    if (
        $temporaryKeyDirectory -and
        (Test-Path -LiteralPath $temporaryKeyDirectory -PathType Container)
    ) {
        Remove-Item -LiteralPath $temporaryKeyDirectory -Force -ErrorAction SilentlyContinue
    }
}
