# Pilar 2B completo: warm-start por imitación + ajuste fino con RL (PBRS activo).
#
#   corrida "sistema"    = pesos de imitación de cada fase (punto de partida)
#   corrida "sistema_ft" = esos pesos después del ajuste fino con RL
#
# PPO parte de los pesos que ya elegiste (persecución BC humano, conducción DAgger
# del experto, definición DAgger humano, jerárquico DAgger humano). A2C y DQN no
# pueden cargar una red PPO, así que primero se clonan con BC sobre los MISMOS CSV
# (que ya contienen las rondas de DAgger agregadas).
#
# Cada algoritmo usa su PROPIO puerto: así los tres pueden correr a la vez en
# terminales distintas sin apagarse el servidor entre ellos.
# Si se corta, vuelve a lanzarlo igual: salta lo terminado y sigue desde checkpoint.pth.
#
# Uso (desde la raíz del proyecto):
#   powershell -ExecutionPolicy Bypass -File scripts/ajuste_fino.ps1 -Algoritmo ppo
#   powershell -ExecutionPolicy Bypass -File scripts/ajuste_fino.ps1 -Algoritmo a2c
#   powershell -ExecutionPolicy Bypass -File scripts/ajuste_fino.ps1 -Algoritmo dqn
param(
    [ValidateSet("ppo", "a2c", "dqn")][string]$Algoritmo = "ppo",
    [int]$Pasos = 50000,            # por habilidad (~3-4 min con PPO)
    [int]$PasosJerarquico = 10000,  # decisiones de la política de coordinación
    [string]$Dispositivo = "cuda",
    [int]$Puerto = 0                # 0: ppo 6600, a2c 6700, dqn 6800
)
$W = "resultados/modelo_weights"
if ($Puerto -eq 0) { $Puerto = @{ ppo = 6600; a2c = 6700; dqn = 6800 }[$Algoritmo] }
$rollout = @{ ppo = 200; a2c = 100; dqn = 1 }[$Algoritmo]

function Correr([string[]]$argumentos) {
    Write-Host "`n>> python proyecto.py $($argumentos -join ' ')" -ForegroundColor Cyan
    & python proyecto.py @argumentos
    if ($LASTEXITCODE -ne 0) { throw "Fallo: python proyecto.py $($argumentos -join ' ')" }
}

function Pasos([string]$archivo) {
    [int](& python -c "import sys, torch; print(int(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['pasos']))" $archivo)
}

# Devuelve los argumentos --continuar/--pasos, o $null si esa fase ya terminó.
function Reanudar([string]$tarea, [int]$total) {
    $base = "$W/$tarea/$Algoritmo/sistema/final.pth"
    $hecho = "$W/$tarea/$Algoritmo/sistema_ft/final.pth"
    $ckpt = "$W/$tarea/$Algoritmo/sistema_ft/checkpoint.pth"
    if (Test-Path $hecho) {
        Write-Host "`n== $tarea/$Algoritmo ya terminado ($hecho); se salta" -ForegroundColor Yellow
        return $null
    }
    if (Test-Path $ckpt) {
        $faltan = $total - ((Pasos $ckpt) - (Pasos $base))
        $faltan = [math]::Max($rollout, [math]::Ceiling($faltan / $rollout) * $rollout)
        Write-Host "`n== $tarea/$Algoritmo sigue desde checkpoint.pth: faltan $faltan pasos" -ForegroundColor Yellow
        return @("--continuar", $ckpt, "--pasos", "$faltan")
    }
    return @("--continuar", $base, "--pasos", "$total")
}

# 1) Warm-start de A2C / DQN (sin servidor; segundos). Se salta si ya existe.
if ($Algoritmo -ne "ppo") {
    $fuentes = @{
        persecucion = @()
        conduccion  = @("--experto", "programado")
        definicion  = @()
        jerarquico  = @()
    }
    foreach ($t in "persecucion", "conduccion", "definicion", "jerarquico") {
        if (Test-Path "$W/$t/$Algoritmo/sistema/final.pth") { continue }
        Correr (@("imitar", "bc", "--tarea", $t, "--algoritmo", $Algoritmo, "--corrida", "sistema",
                "--dispositivo", $Dispositivo, "--epocas-bc", "100") + $fuentes[$t])
    }
}

# Hiperparámetros conservadores: el RL debe mejorar la política clonada, no borrarla.
$extra = switch ($Algoritmo) {
    "ppo" { @("--lr", "0.0001", "--entropia", "0.001", "--kl-objetivo", "0.02") }
    "a2c" { @("--lr", "0.0001", "--entropia", "0.001") }
    "dqn" { @("--lr", "0.0001", "--double", "--epsilon-inicial", "0.1", "--epsilon-final", "0.02", "--warmup", "1000") }
}
$comunes = @("--corrida", "sistema_ft", "--shaping", "pbrs", "--dispositivo", $Dispositivo, "--puerto", "$Puerto") + $extra

# 2) Ajuste fino de las tres habilidades.
foreach ($t in "persecucion", "conduccion", "definicion") {
    $desde = Reanudar $t $Pasos
    if ($desde) { Correr (@("entrenar", "--tarea", $t, "--algoritmo", $Algoritmo) + $desde + $comunes) }
}

# 3) Ajuste fino de la coordinación sobre las habilidades ya ajustadas.
$desde = Reanudar "jerarquico" $PasosJerarquico
if ($desde) {
    Correr (@("entrenar", "--tarea", "jerarquico", "--algoritmo", $Algoritmo) + $desde + @(
            "--habilidades", "$W/persecucion/$Algoritmo/sistema_ft/final.pth",
            "$W/conduccion/$Algoritmo/sistema_ft/final.pth",
            "$W/definicion/$Algoritmo/sistema_ft/final.pth") + $comunes)
}

Write-Host "`nListo: $W/*/$Algoritmo/sistema_ft/final.pth" -ForegroundColor Green
