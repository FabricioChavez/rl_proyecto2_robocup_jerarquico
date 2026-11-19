"""Monitor oficial para los dos controladores de conducción; no entrena."""
from argparse import Namespace
import csv
from datetime import datetime
from pathlib import Path
import pickle
import time
import torch
from entorno.conduccion import ConduccionEnv, comando_parametrizado
from entorno.servidor import local_server
from modelo.conduccion import AgenteConduccion
from modelo.entrenar_conduccion import potencia_entrenamiento
from evaluador.visualizar import official_monitor


def ejecutar(args):
    if not 1 <= args.fps <= 30 or args.episodios < 0:
        raise ValueError('fps entre 1 y 30; episodios >= 0')
    if args.pesos:
        path = Path(args.pesos)
    else:
        if not args.algoritmo or not args.corrida:
            raise ValueError('Indica --pesos o bien --algoritmo y --corrida')
        root = Path('resultados/modelo_weights/conduccion')/args.algoritmo/args.corrida
        path = next((root/name for name in ('checkpoint.pth', 'final.pth', 'mejor.pth')
                     if (root/name).exists()), None)
        if path is None:
            raise FileNotFoundError(f'Todavía no hay checkpoint en {root}')
    # El entrenador puede estar reemplazando checkpoint.pth en este momento.
    for attempt in range(5):
        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            break
        except (EOFError, RuntimeError, pickle.UnpicklingError):
            if attempt == 4:
                raise RuntimeError('No se pudo leer el checkpoint mientras se escribe; vuelve a intentar')
            time.sleep(.5)
    if data.get('formato') != ConduccionEnv.VERSION:
        raise ValueError('Este visualizador necesita pesos de conducción con recuperación')
    spec = Namespace(**data['argumentos'])
    spec.dispositivo = args.dispositivo
    spec.puerto = args.puerto
    torch.set_num_threads(spec.hilos)
    agent = AgenteConduccion(spec)
    agent.restaurar(data, entrenar=False)
    power = args.defensor if args.defensor is not None else potencia_entrenamiento(agent, spec)
    if not 0 <= power <= 100:
        raise ValueError('--defensor debe estar entre 0 y 100')
    output = (Path(args.salida) if args.salida else
              Path('resultados/logs/visualizacion_conduccion')/
              datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'modelo.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['campo', 'valor'])
        for row in [('pesos', str(path)), ('algoritmo', spec.algoritmo), ('pasos', agent.pasos),
                    ('red', str(spec.red)), ('defensor', power), ('metros', spec.metros),
                    ('semilla_inicial', args.semilla), ('aprendizaje', False)]:
            writer.writerow(row)
    print(f'MODELO {spec.algoritmo}; red={spec.red}; pasos={agent.pasos}; pesos={path}', flush=True)
    print(f'Nuestro jugador: AMARILLO 1 (Attack). Defensor azul, potencia={power}. '
          f'Objetivo: conducir {spec.metros} m. Solo inferencia, sin exploración.', flush=True)
    print(f'CSV de acciones y métricas: {output}', flush=True)
    with local_server(args.semilla, args.puerto) as (host, port):
        env = ConduccionEnv(host, port, spec)
        env.defensor = power
        try:
            with official_monitor(args.puerto, output) as gui, (
                output/'trayectoria.csv').open('w', encoding='utf-8', newline='') as stream:
                fields = ['episodio', 'ciclo', 'tipo', 'comando', 'balon_x', 'balon_y',
                          'atacante_x', 'atacante_y', 'distancia', 'avance', 'reward',
                          'retorno', 'recuperando', 'recuperado', 'posesion_rival',
                          'exito', 'motivo']
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                ep = 0
                while gui.poll() is None and (not args.episodios or ep < args.episodios):
                    obs, _ = env.reset(seed=args.semilla+ep)
                    ep += 1
                    print(f'EPISODIO {ep}: inicio_x={env.inicio_x:.2f}; '
                          f'meta_x={env.inicio_x+env.metros:.2f}', flush=True)
                    while gui.poll() is None:
                        _, _, action = agent.actuar(obs, explorar=False)
                        tipo, params = action
                        command = comando_parametrizado(tipo, params)
                        obs, reward, done, _, info = env.step(action)
                        player = env.world.players[('Attack', 1)]
                        writer.writerow(dict(
                            episodio=ep, ciclo=env.t, tipo=('dash','turn','kick')[tipo],
                            comando=command, balon_x=env.world.ball[0], balon_y=env.world.ball[1],
                            atacante_x=player[0], atacante_y=player[1], distancia=info['distancia'],
                            avance=info['avance'], reward=reward, retorno=env.base_return,
                            recuperando=int(info['recuperando']), recuperado=int(info['recuperado']),
                            posesion_rival=info['posesion_rival'], exito=int(info['success']),
                            motivo=info['reason']))
                        stream.flush()
                        if env.t % 5 == 0 or done or info['recuperado']:
                            print(f'ciclo={env.t} acción={command} avance={info["avance"]:.2f}m '
                                  f'distancia={info["distancia"]:.2f}m '
                                  f'recuperando={info["recuperando"]} '
                                  f'retorno={env.base_return:.2f} '
                                  f'exito={info["success"]} motivo={info["reason"]}', flush=True)
                        time.sleep(1/args.fps)
                        if done:
                            break
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
