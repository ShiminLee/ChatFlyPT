param(
    [ValidateRange(1, 8)]
    [int]$Workers = 4,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

# Read-only preflight: use existing local sources and the project's own runtime.
$requiredFiles = @(
    '.venv\Scripts\python.exe',
    'data\raw\malecns-v1.0\body-annotations-male-cns-v1.0-minconf-0.5.feather',
    'data\raw\malecns-v1.0\body-neurotransmitters-male-cns-v1.0.feather',
    'data\raw\malecns-v1.0\connectome-weights-male-cns-v1.0-minconf-0.5.feather',
    'vendor\shiu-brain\Completeness_783.csv',
    'vendor\shiu-brain\Connectivity_783.parquet',
    'vendor\flywire-annotations\supplemental_files\Supplemental_file1_neuron_annotations.tsv',
    'data\derived\reconstruction\stimuli.json',
    'data\derived\reconstruction\decoder.npz'
)
$missingFiles = @($requiredFiles | Where-Object {
    -not (Test-Path -LiteralPath (Join-Path $projectRoot $_) -PathType Leaf)
})
if ($missingFiles.Count -gt 0) {
    throw "Missing local prerequisites; no download or install was attempted:`n$($missingFiles -join "`n")"
}

Push-Location -LiteralPath $projectRoot
try {
    $dependencyCheck = @'
import importlib.util
import sys

required = ['numpy', 'pandas', 'pyarrow', 'scipy', 'numba', 'sklearn',
            'threadpoolctl', 'pytest', 'brian2']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Missing packages in project .venv: ' + ', '.join(missing))
print('Local prerequisites ready; using ' + sys.executable)
'@
    Invoke-CheckedPython -Arguments @('-c', $dependencyCheck)
    if ($CheckOnly) {
        Write-Host 'CheckOnly: no graph build, training, tests, download, or model API call was run.'
        return
    }

    Invoke-CheckedPython -Arguments @('-m', 'chatflypt.social', 'build')
    Invoke-CheckedPython -Arguments @('-m', 'chatflypt.malecns', 'build')
    Invoke-CheckedPython -Arguments @('-m', 'chatflypt.touch_readout', 'train', '--workers', "$Workers")
    # This trainer currently has a fixed four-worker implementation.
    Invoke-CheckedPython -Arguments @('-m', 'chatflypt.social_readout', 'train')
    Invoke-CheckedPython -Arguments @('-m', 'pytest', 'tests', '-q', '--junitxml=reports/full-pathway/pytest.xml')
    Write-Host 'Pathway reproduction finished. Inspect reports/full-pathway/*.json for the current results.'
}
finally {
    Pop-Location
}
