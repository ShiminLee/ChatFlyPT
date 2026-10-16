param([int]$Port = 8765, [switch]$WithLocalModel)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if ($WithLocalModel) {
    Write-Warning 'The fly pathway no longer uses a language model; local inference was not started.'
}
& (Join-Path $projectRoot '.venv\Scripts\python.exe') -m chatflypt.server --port $Port
