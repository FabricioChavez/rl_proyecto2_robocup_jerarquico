param([int]$Pasos=10000,[string]$Escenario='2v1')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force runs | Out-Null
if (Test-Path runs/background.json) {
    $previous=Get-Content runs/background.json | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw "Ya hay un proceso registrado con PID $($previous.pid). Comprueba su estado antes de lanzar otro."
    }
}
$parameters=@('-u','-m','robocup_rl.campana','--steps',"$Pasos",'--seeds','0','1','2','--algorithms','ppo','a2c','dqn','--scenario',$Escenario,'--evaluate','200','--port','6200')
$process=Start-Process -FilePath "$root/.venv/Scripts/python.exe" -ArgumentList $parameters -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root/runs/training.stdout.log" -RedirectStandardError "$root/runs/training.stderr.log" -PassThru
@{pid=$process.Id;started=(Get-Date).ToString('o');scenario=$Escenario;steps_per_skill=$Pasos;arguments=$parameters} | ConvertTo-Json -Depth 5 | Set-Content runs/background.json
Get-Content runs/background.json
