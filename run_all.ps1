param([ValidateSet('auto','cuda','cpu')][string]$Device='auto',[switch]$VerifyOnly)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$Conda='D:\Anaconda\Scripts\conda.exe'
if (-not (Test-Path -LiteralPath $Conda)) { $Conda='conda' }
$EnvironmentJson=(& $Conda env list --json) -join "`n"
if ($LASTEXITCODE -ne 0) { throw 'Cannot list Conda environments' }
$Match=[regex]::Match($EnvironmentJson,'(?s)"envs"\s*:\s*(\[[^\]]*\])')
if (-not $Match.Success) { throw 'Conda returned no environment list' }
$Environments=$Match.Groups[1].Value | ConvertFrom-Json
$Present=$Environments | Where-Object { (Split-Path -Leaf $_) -eq 'bci2a' }
if (-not $Present) {
    & $Conda env create --file environment.yml
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
}
New-Item -ItemType Directory -Path logs -Force | Out-Null
function Invoke-Python {
    & $Conda run --no-capture-output -n bci2a python @args
    if ($LASTEXITCODE -ne 0) { throw "Conda Python exited with $LASTEXITCODE" }
}
& {
    Invoke-Python -m ruff check .
    Invoke-Python -m ruff format --check .
    Invoke-Python -m pytest -q
    if ($VerifyOnly) { Invoke-Python run.py all --device $Device --verify-only }
    else { Invoke-Python run.py all --device $Device }
} *>&1 | Tee-Object -FilePath .\logs\run_all.log
