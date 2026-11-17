param([string]$Subtarea='persecucion',[string]$Algoritmo='ppo',[string]$Modelo,[int]$Episodios=3)
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$arguments=@('-m','robocup_rl.evaluar','--task',$Subtarea,'--algo',$Algoritmo,'--episodes',$Episodios,'--render')
if ($Modelo) { $arguments+=@('--model',$Modelo) }
& .\.venv\Scripts\python.exe @arguments
exit $LASTEXITCODE
