param([ValidateRange(1,6)][int]$Workers=3)
$ErrorActionPreference='Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$taskPython=Join-Path (Get-Location) '.venv\Scripts\python.exe'
function Invoke-Stage([string[]]$StageArgs) {
    & $taskPython -X utf8 @StageArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $StageArgs" }
}
Invoke-Stage @('-c','from chatflypt.sensory_catalogue import build_catalogue; build_catalogue()')
foreach ($dataset in @('flywire','malecns')) {
    Invoke-Stage @('-m','chatflypt.state_readout',$dataset,'--workers',"$Workers")
    Invoke-Stage @('-c',"from chatflypt.state_readout import requalify; requalify('$dataset')")
}
Invoke-Stage @('-m','chatflypt.gap_audit')
Invoke-Stage @('-m','chatflypt.state_fresh_check')
Invoke-Stage @('-m','pytest','tests','-q','--junitxml=reports/gap-closure/pytest.xml')
Invoke-Stage @('reports/body-simulation/reproduce.py')
Invoke-Stage @('-m','chatflypt.gap_report')
