param([string]$Tarea='global',[string]$Algoritmo='ppo',[int]$Semilla=0,[string]$Escenario='2v1',[string]$Configuracion='')
$ErrorActionPreference='Stop'
Set-Location $PSScriptRoot
$parameters=@('-m','robocup_rl.monitor','--task',$Tarea,'--algo',$Algoritmo,'--seed',"$Semilla",'--scenario',$Escenario,'--port','6500')
if ($Configuracion) { $parameters+=@('--scene',$Configuracion) }
& .\.venv\Scripts\python.exe @parameters
exit $LASTEXITCODE
