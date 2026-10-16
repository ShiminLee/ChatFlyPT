param([ValidateRange(1024,65535)][int]$Port = 8081,
      [Alias('Model')][ValidateSet('1.7b','4b')][string]$ModelPreset = '4b')
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBase = Join-Path $taskRoot 'reports/local-model-2026-09-19'
$taskRuntime = Join-Path $taskBase 'runtime'
$taskExecutable = Join-Path $taskRuntime 'llama-server.exe'
$taskModelFile = if ($ModelPreset -eq '4b') { 'qwen3-4b-instruct-2507-q8_0.gguf' } else { 'Qwen3-1.7B-Q8_0.gguf' }
$taskAlias = if ($ModelPreset -eq '4b') { 'chatflypt-qwen3-4b-instruct-2507' } else { 'chatflypt-qwen3-1.7b' }
$taskModel = Join-Path $taskBase ('models/' + $taskModelFile)
if (!(Test-Path -LiteralPath $taskExecutable) -or !(Test-Path -LiteralPath $taskModel)) {
    throw 'Run scripts/local-model-install.py first.'
}
function Save-LocalModelConfig {
    $taskConfigDir = Join-Path $taskRoot 'data/local'
    New-Item -ItemType Directory -Path $taskConfigDir -Force | Out-Null
    @{base_url="http://127.0.0.1:$Port/v1";model=$taskAlias} | ConvertTo-Json |
        Set-Content (Join-Path $taskConfigDir 'local-model.json') -Encoding utf8
}
$taskListeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
if ($taskListeners.Count) {
    $taskOwned = $true
    foreach ($taskListener in $taskListeners) {
        $taskExisting = Get-Process -Id $taskListener.OwningProcess -ErrorAction SilentlyContinue
        if (!$taskExisting -or $taskExisting.Path -ne $taskExecutable) { $taskOwned = $false }
    }
    if ($taskOwned) {
        try {
            $taskHealth = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 3
            $taskModels = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 3
            if ($taskHealth.status -eq 'ok' -and $taskAlias -in $taskModels.data.id) {
                Save-LocalModelConfig
                Write-Output "Already running this project's $taskAlias at http://127.0.0.1:$Port/v1"
                exit 0
            }
        } catch { }
    }
    throw "Port $Port is occupied by a different or unavailable service/model; no existing process was stopped."
}
$taskArgs = @('--model', $taskModel, '--alias', $taskAlias,
    '--host', '127.0.0.1', '--port', "$Port", '--ctx-size', '4096', '--parallel', '1',
    '--threads', '8', '--n-gpu-layers', '99', '--flash-attn', 'on',
    '--reasoning', 'off', '--reasoning-budget', '0', '--temp', '0.7',
    '--top-p', '0.8', '--top-k', '20', '--min-p', '0', '--presence-penalty', '1.5',
    '--seed', '84217', '--no-webui', '--cors-origins', "http://127.0.0.1:$Port")
$taskProcess = Start-Process -FilePath $taskExecutable -ArgumentList $taskArgs -WorkingDirectory $taskRuntime `
    -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskBase 'server-stdout.log') `
    -RedirectStandardError (Join-Path $taskBase 'server-stderr.log')
@{pid=$taskProcess.Id; executable=$taskExecutable; arguments=$taskArgs;
    base_url="http://127.0.0.1:$Port/v1"; model=$taskAlias;
    started_at=(Get-Date -Format o); window='Hidden'; host_scope='loopback-only';
    production_request_parameters='messages,max_completion_tokens=1024,stream=false,store=false; no per-request benchmark-only constraints'
} | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $taskBase 'runtime.json') -Encoding utf8
$taskDeadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $taskDeadline) {
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) { throw "Local model exited; inspect reports/local-model-2026-09-19/server-stderr.log" }
    try {
        $taskHealth = Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2
        if ($taskHealth.status -eq 'ok') {
            $taskModels = Invoke-RestMethod "http://127.0.0.1:$Port/v1/models" -TimeoutSec 2
            if ($taskAlias -in $taskModels.data.id) {
                Save-LocalModelConfig
                Write-Output "Ready: $taskAlias, process $($taskProcess.Id), http://127.0.0.1:$Port/v1"
                exit 0
            }
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw 'Local model health was not ready within 90 seconds; local config was not changed.'
