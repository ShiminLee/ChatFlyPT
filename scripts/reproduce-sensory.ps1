param([int]$Workers = 2)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Install the project Python environment first.' }
& $taskPython -m chatflypt.sensory_catalogue build
if ($LASTEXITCODE -ne 0) { throw 'Sensory catalogue build failed' }
& $taskPython -m chatflypt.sensory_readout flywire --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'FlyWire readout failed' }
& $taskPython -m chatflypt.sensory_readout malecns --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'MaleCNS readout failed' }
& $taskPython -m chatflypt.sensory_audit run --workers $Workers
if ($LASTEXITCODE -ne 0) { throw 'Sensory delivery audit failed' }
& $taskPython -m chatflypt.sensory_fresh_check
if ($LASTEXITCODE -ne 0) { throw 'Fresh neural checks failed' }
& $taskPython -m pytest tests -q --junitxml=reports/general-sensory/pytest.xml
if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
& $taskPython -m chatflypt.sensory_report
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
