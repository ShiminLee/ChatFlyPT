param([string]$PythonPath)
# Builds .venv-body: the isolated physics environment (FlyGym 2.1.0 / MuJoCo 3.9.0,
# Python 3.12) used by chatflypt.behavior_body's worker subprocess.
# Requires vendor/flygym to exist first — run scripts/acquire_sources.py before this.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath 'vendor\flygym')) {
    throw 'vendor\flygym is missing. Run: .venv\Scripts\python.exe scripts\acquire_sources.py'
}
if (-not (Test-Path -LiteralPath '.venv-body\Scripts\python.exe')) {
    if ($PythonPath) {
        & $PythonPath -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 required"'
        if ($LASTEXITCODE -ne 0) { throw 'Use a Python 3.12 executable.' }
        & $PythonPath -m venv .venv-body
    } else {
        py -3.12 -m venv .venv-body
    }
    if ($LASTEXITCODE -ne 0) { throw 'Create the environment with Python 3.12 or pass -PythonPath.' }
}
& '.\.venv-body\Scripts\python.exe' -m pip install -r 'research\requirements-body.lock.txt'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
# FlyGym is pinned as an editable install of the vendored source; its dependencies
# are already fixed by the lock file, so --no-deps keeps the resolution identical.
& '.\.venv-body\Scripts\python.exe' -m pip install -e 'vendor\flygym' --no-deps
if ($LASTEXITCODE -ne 0) { throw 'flygym editable install failed.' }
& '.\.venv-body\Scripts\python.exe' -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
# WMI shim: on hosts with a degraded WMI service, platform.uname() aborts the
# process at mujoco import time (0x8007000e) before any project code runs.
# The shim reads PROCESSOR_* environment variables instead; see the negative
# registry entry "degraded-wmi-aborts-the-body-worker".
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'body-env-sitecustomize.py') `
    -Destination '.\.venv-body\Lib\site-packages\sitecustomize.py' -Force
& '.\.venv-body\Scripts\python.exe' -c "import flygym, mujoco; print('flygym', flygym.__version__ if hasattr(flygym,'__version__') else '2.1.0', '| mujoco', mujoco.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Body environment self-check failed.' }
