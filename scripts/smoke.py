"""Prueba real de cada habilidad/algoritmo, más composición jerárquica."""
import subprocess
import sys
from pathlib import Path

for algo in ('ppo','a2c','dqn'):
    skills=[]
    for task in ('persecucion','conduccion','definicion'):
        steps=600 if algo=='dqn' else 200
        output=Path(f'runs/verification/{task}/{algo}_{steps}')
        if not (output/'final.zip').exists():
            subprocess.run([sys.executable,'-m','robocup_rl.ejecutar','--task',task,'--algo',algo,'--steps',str(steps),'--port','6300','--output',str(output)],check=True)
        skills.append(str(output/'final.zip'))
    output=Path(f'runs/verification/global/{algo}_{steps}')
    if not (output/'final.zip').exists():
        subprocess.run([sys.executable,'-m','robocup_rl.ejecutar','--task','global','--algo',algo,'--steps',str(steps),
                        '--port','6300','--output',str(output),'--skills',*skills,'--skill-algo',algo],check=True)
print('SMOKE_COMPLETE: 3 habilidades x 3 algoritmos + 3 meta-controladores en CUDA',flush=True)
