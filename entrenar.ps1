param([string]$Subtarea='persecucion',[string]$Algoritmo='ppo',[int]$Pasos=10000,[int]$Semilla=0)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m robocup_rl.ejecutar --task $Subtarea --algo $Algoritmo --steps $Pasos --seed $Semilla --port 6100
exit $LASTEXITCODE
