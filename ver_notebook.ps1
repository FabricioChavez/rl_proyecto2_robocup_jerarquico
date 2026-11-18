$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m jupyterlab notebooks/visualizacion_p2_jeffrey.ipynb
exit $LASTEXITCODE
