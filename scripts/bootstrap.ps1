param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($PythonPath) {
        & $PythonPath -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
        if ($LASTEXITCODE -ne 0) { throw 'Use a Python 3.12 executable.' }
        & $PythonPath -m venv .venv
    } else {
        py -3.12 -m venv .venv
    }
    if ($LASTEXITCODE -ne 0) { throw 'Create the environment with Python 3.12 or pass -PythonPath.' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r 'research\requirements-baseline.lock.txt'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& '.\.venv\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
