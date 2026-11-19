"""CLI y cola de procesos de conducción. Sin librerías de algoritmos RL."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ALGORITMOS = ('ddqn_dueling', 'mpdqn')


def opciones(p):
    p.add_argument('--dispositivo', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--pasos', type=int, default=100000)
    p.add_argument('--puerto', type=int, default=6700)
    p.add_argument('--horizonte', type=int, default=300)
    p.add_argument('--metros', type=float, default=30.)
    p.add_argument('--radio-control', type=float, default=1.085)
    p.add_argument('--distancia-maxima', type=float, default=3.)
    p.add_argument('--gracia-control', type=int, default=12)
    p.add_argument('--gracia-recuperacion', type=int, default=40)
    p.add_argument('--distancia-recuperacion', type=float, default=12.)
    p.add_argument('--posesion-rival-maxima', type=int, default=12)
    p.add_argument('--costo-sin-control', type=float, default=-.1)
    p.add_argument('--defensor', type=int, default=100)
    p.add_argument('--arquero', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--curriculo', nargs='*', default=['0:0', '20000:40', '60000:100'],
                   help='Lista INICIO:POTENCIA_DEFENSOR; sin valores: potencia fija')
    p.add_argument('--modo-curriculo', choices=['rendimiento', 'pasos'], default='rendimiento')
    p.add_argument('--umbral-curriculo', type=float, default=.8)
    p.add_argument('--ventana-curriculo', type=int, default=50)
    p.add_argument('--min-episodios-curriculo', type=int, default=50)
    p.add_argument('--premio', type=float, default=100.)
    p.add_argument('--perdida', type=float, default=-30.)
    p.add_argument('--intercepcion', type=float, default=-50.)
    p.add_argument('--costo', type=float, default=-.2)
    p.add_argument('--pbrs', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--escala-pbrs', type=float, default=1.)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-actor', type=float, default=1e-4)
    p.add_argument('--tau', type=float, default=.005)
    p.add_argument('--norma-gradiente', type=float, default=10.)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--buffer', type=int, default=100000)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--train-freq', type=int, default=4)
    p.add_argument('--gradient-steps', type=int, default=1)
    p.add_argument('--epsilon-inicial', type=float, default=1.)
    p.add_argument('--epsilon-final', type=float, default=.05)
    p.add_argument('--epsilon-pasos', type=int, default=60000)
    p.add_argument('--checkpoint-cada', type=int, default=2000)
    p.add_argument('--evaluar-cada', type=int, default=10000,
                   help='0: evaluar solo al terminar')
    p.add_argument('--episodios-validacion', type=int, default=50)
    p.add_argument('--semilla-validacion', type=int, default=20000)
    p.add_argument('--log-cada', type=int, default=500)
    p.add_argument('--hilos', type=int, default=2)


def registrar(commands):
    train = commands.add_parser('entrenar-conduccion', help='Conducción v2: Double+Dueling o MP-DQN propio')
    opciones(train)
    train.add_argument('--algoritmo', choices=ALGORITMOS, required=True)
    train.add_argument('--red', type=int, nargs='+', default=[128, 128])
    train.add_argument('--semilla', type=int, default=0)
    train.add_argument('--corrida', required=True)
    train.add_argument('--continuar', help='PTH v2; --pasos son decisiones adicionales')
    batch = commands.add_parser('lote-conduccion', help='Cola de algoritmos × redes × semillas')
    opciones(batch)
    batch.add_argument('--algoritmos', choices=ALGORITMOS, nargs='+', default=list(ALGORITMOS))
    batch.add_argument('--redes', nargs='+', default=['128x128'])
    batch.add_argument('--semillas', type=int, nargs='+', default=[0])
    batch.add_argument('--lote', required=True)
    batch.add_argument('--paralelos', type=int, default=1)
    batch.add_argument('--detach', action='store_true')
    batch.add_argument('--vista-previa', action='store_true')
    evaluation = commands.add_parser('evaluar-conduccion', help='Evaluación independiente de un PTH v2')
    evaluation.add_argument('--pesos', required=True)
    evaluation.add_argument('--dispositivo', choices=['cuda', 'cpu'], default='cuda')
    evaluation.add_argument('--episodios', type=int, default=200)
    evaluation.add_argument('--semilla-prueba', type=int, default=10000)
    evaluation.add_argument('--puerto', type=int, default=6800)
    evaluation.add_argument('--defensor', type=int)
    evaluation.add_argument('--salida')
    graph = commands.add_parser('graficar-conduccion', help='PNG desde los CSV nuevos; sin interfaz')
    graph.add_argument('--ventana', type=int, default=50)
    visual = commands.add_parser('ver-conduccion', help='Ver un checkpoint nuevo en rcssmonitor oficial')
    visual.add_argument('--pesos')
    visual.add_argument('--algoritmo', choices=ALGORITMOS)
    visual.add_argument('--corrida')
    visual.add_argument('--dispositivo', choices=['cpu', 'cuda'], default='cpu')
    visual.add_argument('--defensor', type=int, help='Por defecto: etapa del checkpoint')
    visual.add_argument('--puerto', type=int, default=6900)
    visual.add_argument('--semilla', type=int, default=10000)
    visual.add_argument('--episodios', type=int, default=0, help='0: repetir hasta cerrar')
    visual.add_argument('--fps', type=float, default=10)
    visual.add_argument('--salida')


def validar(args):
    import re
    for name in ('corrida', 'lote'):
        value = getattr(args, name, None)
        if value and not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError(f'--{name}: usa letras, números, guion o subrayado')
    for name in ('pasos', 'horizonte', 'gracia_control', 'gracia_recuperacion',
                 'posesion_rival_maxima', 'ventana_curriculo', 'min_episodios_curriculo',
                 'batch', 'buffer', 'train_freq',
                 'gradient_steps', 'epsilon_pasos', 'checkpoint_cada',
                 'episodios_validacion', 'log_cada', 'hilos', 'episodios', 'paralelos', 'ventana'):
        if name == 'episodios' and args.command == 'ver-conduccion' and args.episodios == 0:
            continue
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f'{name} debe ser positivo')
    if hasattr(args, 'gamma'):
        if not 0 < args.gamma <= 1 or not 0 < args.tau <= 1:
            raise ValueError('gamma y tau deben estar en (0,1]')
        if not 0 < args.metros < 40:
            raise ValueError('--metros debe estar entre 0 y 40 desde este inicio de campo')
        if not 0 < args.radio_control < args.distancia_maxima:
            raise ValueError('radio-control debe ser positivo y menor que distancia-maxima')
        if args.distancia_recuperacion < args.distancia_maxima or args.gracia_recuperacion < args.gracia_control:
            raise ValueError('La recuperación no puede ser menor que la tolerancia normal de control')
        if not 0 < args.umbral_curriculo <= 1 or args.costo_sin_control > 0:
            raise ValueError('Umbral de currículo en (0,1] y costo-sin-control <= 0')
        if args.lr <= 0 or args.lr_actor <= 0 or args.norma_gradiente <= 0:
            raise ValueError('Tasas de aprendizaje y norma deben ser positivas')
        if not 0 <= args.epsilon_final <= args.epsilon_inicial <= 1:
            raise ValueError('Necesita 0 <= epsilon-final <= epsilon-inicial <= 1')
        if args.warmup < 0 or args.evaluar_cada < 0 or args.buffer < args.batch:
            raise ValueError('Warmup/evaluar-cada no negativos; buffer >= batch')
        if getattr(args, 'red', None) and any(n <= 0 for n in args.red):
            raise ValueError('Tamaños de red positivos')
        starts = []
        for item in args.curriculo:
            inicio, power = map(int, item.split(':'))
            if inicio < 0 or not 0 <= power <= 100:
                raise ValueError('Currículo: INICIO>=0 y potencia 0..100')
            starts.append(inicio)
        if starts and (starts[0] != 0 or starts != sorted(set(starts))):
            raise ValueError('El currículo empieza en 0 y tiene inicios crecientes únicos')
    if getattr(args, 'defensor', None) is not None and not 0 <= args.defensor <= 100:
        raise ValueError('Potencia del defensor: 0..100')


def graficar(args):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    total = 0
    for algo in ALGORITMOS:
        root = Path('resultados/logs/conduccion')/algo
        for folder in root.glob('*'):
            episodes = folder/'episodios.csv'
            if not episodes.exists():
                continue
            f = pd.read_csv(episodes)
            if f.empty:
                continue
            destination = Path('resultados/graficas/conduccion_v2')/algo/folder.name
            destination.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for ax, column, title in zip(axes.flat[:3], ['retorno', 'exito', 'avance'],
                                        ['Retorno base (entrenamiento)', 'Éxito (entrenamiento)', 'Avance (m)']):
                ax.plot(f.pasos, f[column].rolling(args.ventana, min_periods=1).mean())
                ax.set_title(title); ax.set_xlabel('Decisiones'); ax.grid(alpha=.2)
            validation = folder/'validacion.csv'
            if validation.exists():
                v = pd.read_csv(validation)
                axes[1, 1].plot(v.pasos, v.exito, marker='o')
            axes[1, 1].set(title='Validación: defensor objetivo fijo', ylim=(-.02, 1.02), xlabel='Decisiones')
            fig.suptitle(f'{algo} / {folder.name} — currículo cambia dificultad del entrenamiento')
            fig.tight_layout(); fig.savefig(destination/'aprendizaje.png', dpi=150); plt.close(fig)
            updates = folder/'actualizaciones.csv'
            if updates.exists():
                u = pd.read_csv(updates)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                for ax, name in zip(axes, ('perdida_q', 'perdida_actor')):
                    ax.plot(u.pasos, u[name]); ax.set_title(name); ax.grid(alpha=.2)
                fig.tight_layout(); fig.savefig(destination/'perdidas.png', dpi=150); plt.close(fig)
            total += 1
    print(f'Gráficas de {total} corridas: resultados/graficas/conduccion_v2/', flush=True)


def lote(args):
    import torch
    from modelo.pesos import dispositivo_elegido
    dispositivo_elegido(args.dispositivo)  # Error claro, sin fallback a CPU.
    jobs = []
    for algorithm in args.algoritmos:
        for network in args.redes:
            try:
                layers = [int(value) for value in network.split('x')]
                if not layers or any(value <= 0 for value in layers):
                    raise ValueError()
            except ValueError:
                raise ValueError('--redes usa tamaños como 128x128 256x128')
            for seed in args.semillas:
                jobs.append((algorithm, layers, seed, f'{args.lote}_{network}_s{seed}'))
    jobs.sort(key=lambda job: (args.redes.index('x'.join(map(str, job[1]))),
                              job[2], args.algoritmos.index(job[0])))
    if args.puerto < 1024 or args.puerto+3*max(0, args.paralelos-1)+2 > 65535:
        raise ValueError('Rango de puertos inválido')
    for index, job in enumerate(jobs):
        print(index+1, *job, flush=True)
    if args.vista_previa:
        return
    root = Path('resultados/logs/lotes')/args.lote
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        if Path('/.dockerenv').exists():
            raise ValueError('En Docker usa docker compose run -d; omite --detach interno')
        pidfile = root/'lote.pid'
        if pidfile.exists() and os.name == 'nt':
            import ctypes
            pid = int(pidfile.read_text().strip())
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                raise ValueError('Ese lote todavía tiene un proceso activo; revisa lote.pid')
        argv = [sys.executable, '-u', str(Path('proyecto.py').resolve()),
                *[value for value in sys.argv[1:] if value != '--detach']]
        with (root/'lote.log').open('a', encoding='utf-8') as stream:
            kwargs = ({'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
                      if os.name == 'nt' else {'start_new_session': True})
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, cwd=Path.cwd(), **kwargs)
        pidfile.write_text(str(process.pid), encoding='ascii')
        print(f'DETACH PID={process.pid}; progreso: {root}/lote.log; jobs: {root}/jobs/', flush=True)
        return
    (root/'jobs').mkdir(exist_ok=True)
    lock = threading.Lock()
    free_slots = list(range(args.paralelos))
    excluded = {'command', 'algoritmos', 'redes', 'semillas', 'lote', 'paralelos', 'detach', 'vista_previa'}
    common = []
    for key, value in vars(args).items():
        if key in excluded or key == 'puerto':
            continue
        flag = '--'+key.replace('_', '-')
        if isinstance(value, bool):
            common.append(flag if value else '--no-'+key.replace('_', '-'))
        elif isinstance(value, list):
            common.extend([flag, *map(str, value)])
        elif value is not None:
            common.extend([flag, str(value)])

    def run(job):
        from modelo.entrenar_conduccion import csv_row
        algorithm, layers, seed, name = job
        with lock:
            slot = free_slots.pop(0)
        port = args.puerto+3*slot
        folder = Path('resultados/modelo_weights/conduccion')/algorithm/name
        resume = folder/'checkpoint.pth'
        final = folder/'final.pth'
        remaining = args.pasos
        if final.exists():
            resume = final
        if resume.exists():
            data = torch.load(resume, map_location='cpu', weights_only=True)
            from entorno.conduccion import ConduccionEnv
            if data.get('formato') != ConduccionEnv.VERSION:
                raise ValueError('Ese lote usa reglas antiguas; elige un --lote nuevo para recuperación')
            remaining = max(0, args.pasos-data['pasos'])
        else:
            resume = None
        if not remaining:
            print(f'OMITIDO {name}/{algorithm}: presupuesto terminado', flush=True)
            with lock:
                free_slots.append(slot)
            return 0
        argv = [sys.executable, '-u', str(Path('proyecto.py').resolve()), 'entrenar-conduccion',
                *common, '--algoritmo', algorithm, '--red', *map(str, layers),
                '--semilla', str(seed), '--corrida', name, '--puerto', str(port),
                '--pasos', str(remaining)]
        if resume:
            argv.extend(['--continuar', str(resume)])
        logfile = root/'jobs'/f'{algorithm}_{name}.log'
        with lock:
            csv_row(root/'estado.csv', dict(tiempo=time.time(), algoritmo=algorithm,
                    corrida=name, estado='iniciado', codigo='', log=str(logfile)))
        print(f'INICIADO {algorithm}/{name}; puerto={port}; log={logfile}', flush=True)
        try:
            with logfile.open('a', encoding='utf-8') as stream:
                result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            with lock:
                csv_row(root/'estado.csv', dict(tiempo=time.time(), algoritmo=algorithm,
                        corrida=name, estado='terminado' if result.returncode==0 else 'error',
                        codigo=result.returncode, log=str(logfile)))
            print(f'FIN {algorithm}/{name}: código={result.returncode}', flush=True)
            return result.returncode
        finally:
            with lock:
                free_slots.append(slot)
    with ThreadPoolExecutor(max_workers=args.paralelos) as pool:
        results = list(pool.map(run, jobs))
    print(f'LOTE terminado: {sum(code==0 for code in results)}/{len(results)} sin error', flush=True)
    if any(results):
        raise RuntimeError(f'Hay entrenamientos fallidos: consulta {root}/estado.csv y jobs/')


def ejecutar(args):
    validar(args)
    if args.command == 'lote-conduccion':
        return lote(args)
    if args.command == 'entrenar-conduccion':
        from modelo.entrenar_conduccion import entrenar
        return entrenar(args)
    if args.command == 'evaluar-conduccion':
        from modelo.entrenar_conduccion import evaluar_checkpoint
        return evaluar_checkpoint(args)
    if args.command == 'ver-conduccion':
        from evaluador.visualizar_conduccion import ejecutar as visualizar
        return visualizar(args)
    return graficar(args)
