param([switch]$SkipLLM, [int]$Workers = 4)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Run scripts/bootstrap.ps1 first, then install research/requirements-reconstruction.lock.txt.' }
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed: $($Arguments -join ' ')" }
}
Invoke-CheckedPython -Arguments @('-m','chatflypt.verify')
Invoke-CheckedPython -Arguments @('-m','chatflypt.experiment','verify-full')
Invoke-CheckedPython -Arguments @('-m','chatflypt.experiment','physiology')
Invoke-CheckedPython -Arguments @('-m','chatflypt.experiment','generate','--workers',"$Workers")
Invoke-CheckedPython -Arguments @('-m','chatflypt.experiment','train')
Invoke-CheckedPython -Arguments @('-m','chatflypt.demo','--workers',"$Workers")
Invoke-CheckedPython -Arguments @('-m','pytest','tests','-q')
Invoke-CheckedPython -Arguments @('-m','chatflypt.report')
