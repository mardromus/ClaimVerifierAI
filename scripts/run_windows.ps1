# ClaimVerifier AI: set up (first run only) and start the web app on Windows, x64 or ARM64.
#
#   run_windows.bat                                  (double-click, or from a terminal)
#   run_windows.bat -Config configs/lite.yaml        (offline pipeline, no model downloads)
#   run_windows.bat -Explanation ollama              (LLM explanations from a local Ollama server)
#
# Every step is skipped when its result already exists, so later runs start the server directly.

param(
    [string]$Config = "configs/verisci.yaml",
    [string]$Explanation = "template",
    [int]$Port = 8000,
    [switch]$NoBrowser
)

# Native commands are checked through $LASTEXITCODE (Windows PowerShell turns their stderr into errors otherwise).
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($text) { Write-Host ""; Write-Host "==> $text" -ForegroundColor Cyan }

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

$IsArm = ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") -or ($env:PROCESSOR_ARCHITEW6432 -eq "ARM64")

function Find-Python {
    # Python 3.12 / 3.13 / 3.11 registered with the py launcher; the native build first on ARM64.
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) { return $null }
    $lines = @(& py -0p 2>&1 | ForEach-Object { "$_" })
    $found = @()
    foreach ($line in $lines) {
        # " -V:3.12-arm64 *  C:\...\python.exe" (newer launchers) or " -3.12-64  C:\...\python.exe" (older ones)
        if ($line -match '-(?:V:)?(3\.1[123])(-arm64|-64|-32)?\s+\*?\s*(\S.*python\.exe)\s*$') {
            $found += [pscustomobject]@{ Version = $Matches[1]; Tag = "$($Matches[2])"; Path = $Matches[3].Trim() }
        }
    }
    foreach ($v in "3.12", "3.13", "3.11") {
        $match = $found | Where-Object { $_.Version -eq $v }
        if ($IsArm) { $native = $match | Where-Object { $_.Tag -eq "-arm64" }; if ($native) { return $native[0].Path } }
        elseif ($match) { return $match[0].Path }
    }
    if ($found) { return $found[0].Path }
    return $null
}

# ---------------------------------------------------------------- Python environment
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Step "Creating a Python virtual environment in .venv"
    $Base = Find-Python
    $WantNative = $IsArm -and (-not $Base -or $Base -notmatch 'arm64')
    if ((-not $Base -or $WantNative) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Step "Installing Python 3.12 for this user with winget"
        $wingetArgs = @("install", "-e", "--id", "Python.Python.3.12", "--scope", "user", "--silent",
                        "--accept-package-agreements", "--accept-source-agreements")
        if ($IsArm) { $wingetArgs += @("--architecture", "arm64") }
        & winget @wingetArgs
        $Base = Find-Python
        if (-not $Base) {
            $guess = @("$env:LOCALAPPDATA\Programs\Python\Python312-arm64\python.exe",
                       "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe") | Where-Object { Test-Path $_ }
            if ($guess) { $Base = $guess[0] }
        }
    }
    if (-not $Base) { throw "Python 3.11-3.13 not found. Install Python 3.12 from https://www.python.org/downloads/windows/ and run this again." }
    Write-Host "Using $Base"
    Invoke-Checked $Base @("-m", "venv", ".venv")
}

$ReqHash = (Get-FileHash requirements.txt -Algorithm SHA256).Hash
$Marker = Join-Path $Root ".venv\.requirements.sha256"
if (-not (Test-Path $Marker) -or (Get-Content $Marker -Raw).Trim() -ne $ReqHash) {
    Step "Installing Python packages (first run: a few minutes)"
    Invoke-Checked $Py @("-m", "pip", "install", "--upgrade", "pip")
    # PyTorch publishes its Windows ARM64 CPU wheels only on its own index; on x64 the index is harmless.
    Invoke-Checked $Py @("-m", "pip", "install", "-r", "requirements.txt",
                         "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    Set-Content -Path $Marker -Value $ReqHash -Encoding ascii
}

# ---------------------------------------------------------------- data, index and models
$Artifacts = (& $Py -c "from claimverifier.config import load_config; print(load_config(r'$Config').artifacts_dir)").Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not read $Config" }
if (-not (Test-Path (Join-Path $Artifacts "lite\rationale_features.joblib"))) {
    Step "Downloading SciFact and building the search index (about a minute)"
    Invoke-Checked $Py @("-m", "claimverifier", "setup", "--config", $Config, "--no-calibrate")
}
if ($Config -match "verisci" -and -not (Test-Path "models\verisci-roberta-large\config.json")) {
    Step "Downloading the RoBERTa-large verifier (1.3 GB, once)"
    Invoke-Checked $Py @("-m", "claimverifier", "fetch-verifier")
}
if ($Config -notmatch "lite" -and -not (Test-Path "models\scibert-rationale\config.json")) {
    Write-Host ""
    Write-Host "Note: no trained SciBERT evidence selector in models\scibert-rationale, so a simpler fallback is used." -ForegroundColor Yellow
    Write-Host "      For the full accuracy, run once (about an hour):" -ForegroundColor Yellow
    Write-Host "      .venv\Scripts\python -m claimverifier train-rationale --config $Config --neg-ratio 4 --lr 3e-5" -ForegroundColor Yellow
}

# ---------------------------------------------------------------- serve
$Url = "http://localhost:$Port"
if (-not $NoBrowser) {
    # Open the browser as soon as the models are loaded and the server answers.
    Start-Job -ArgumentList $Url -ScriptBlock {
        param($u)
        for ($i = 0; $i -lt 300; $i++) {
            try { Invoke-WebRequest -UseBasicParsing "$u/api/health" -TimeoutSec 2 | Out-Null; Start-Process $u; break }
            catch { Start-Sleep -Seconds 2 }
        }
    } | Out-Null
}
Step "Starting ClaimVerifier AI on $Url (loading models takes a moment; press Ctrl+C to stop)"
& $Py -m claimverifier serve --config $Config --port $Port --set "explanation.backend=$Explanation"
