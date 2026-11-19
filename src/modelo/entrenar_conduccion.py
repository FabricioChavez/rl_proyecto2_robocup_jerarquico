"""Loop propio de conducción, CSV, validación separada y checkpoints PTH."""
import csv
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import torch
from entorno.servidor import local_server
from entorno.conduccion import ConduccionEnv
from modelo.conduccion import AgenteConduccion


def csv_row(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def curriculum(args, paso):
    power = args.defensor if not args.curriculo else int(args.curriculo[0].split(':')[1])
    for token in args.curriculo:
        inicio, valor = map(int, token.split(':'))
        if paso >= inicio:
            power = valor
    return power


def potencia_entrenamiento(agent, args):
    if args.modo_curriculo == 'pasos' or not args.curriculo:
        return curriculum(args, agent.pasos)
    niveles = [int(token.split(':')[1]) for token in args.curriculo]
    return niveles[min(agent.curriculum['indice'], len(niveles)-1)]


def registrar_rendimiento(agent, args, exito):
    if args.modo_curriculo != 'rendimiento' or not args.curriculo:
        return
    state = agent.curriculum
    state['ventana'].append(int(exito))
    state['ventana'] = state['ventana'][-args.ventana_curriculo:]
    state['episodios_etapa'] += 1
    if (state['indice'] < len(args.curriculo)-1
            and state['episodios_etapa'] >= args.min_episodios_curriculo
            and len(state['ventana']) == args.ventana_curriculo
            and float(np.mean(state['ventana'])) >= args.umbral_curriculo):
        tasa = float(np.mean(state['ventana']))
        state['indice'] += 1
        state['ventana'] = []
        state['episodios_etapa'] = 0
        print(f'CURRICULO: éxito reciente={tasa:.1%}; '
              f'nueva potencia del defensor={potencia_entrenamiento(agent, args)}', flush=True)


def evaluar_agente(agent, env, args, episodios, semilla, path=None):
    """Sin exploración ni actualizaciones; nunca usa el éxito fácil del currículo."""
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if agent.device.type == 'cuda' else []
    env_rng = deepcopy(env.np_random.bit_generator.state)
    previous_power = env.defensor
    env.defensor = args.defensor
    rows = []
    try:
        for ep in range(episodios):
            observation, _ = env.reset(seed=semilla+ep)
            total = 0.
            recuperaciones = 0
            while True:
                _, _, action = agent.actuar(observation, explorar=False)
                observation, reward, done, _, info = env.step(action)
                total += args.gamma**(env.t-1)*reward
                recuperaciones += int(info['recuperado'])
                if done:
                    break
            row = dict(episodio=ep, semilla=semilla+ep, pasos_modelo=agent.pasos,
                       algoritmo=args.algoritmo, retorno=env.base_return,
                       retorno_shaping=total, exito=int(info['success']),
                       avance=info['avance'], ciclos=info['ciclos'], motivo=info['reason'],
                       defensor=args.defensor, arquero=int(args.arquero), metros=args.metros,
                       recuperaciones=recuperaciones)
            rows.append(row)
            if path is not None:
                csv_row(path, row)
    finally:
        env.defensor = previous_power
        env.np_random.bit_generator.state = env_rng
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state:
            torch.cuda.set_rng_state_all(cuda_state)
    return dict(episodios=len(rows), exito=float(np.mean([row['exito'] for row in rows])),
                retorno=float(np.mean([row['retorno'] for row in rows])),
                avance=float(np.mean([row['avance'] for row in rows])))


def entrenar(args):
    torch.set_num_threads(args.hilos)
    agent = AgenteConduccion(args)
    folder = Path('resultados/modelo_weights/conduccion')/args.algoritmo/args.corrida
    logs = Path('resultados/logs/conduccion')/args.algoritmo/args.corrida
    folder.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    if (folder/'final.pth').exists() and not args.continuar:
        raise FileExistsError(f'{folder}/final.pth ya existe; usa otra --corrida o --continuar')
    if args.continuar:
        data = torch.load(args.continuar, map_location='cpu', weights_only=True)
        agent.restaurar(data)
    objetivo = agent.pasos+args.pasos
    best_path = folder/'mejor.pth'
    best_score = (-1., -float('inf'))
    if best_path.exists():
        best_data = torch.load(best_path, map_location='cpu', weights_only=True)
        saved = best_data.get('validacion')
        if saved:
            best_score = (saved['exito'], saved['avance'])
    start = time.monotonic()
    last = dict(perdida_q='', perdida_actor='', q_media='', objetivo_td='')
    next_save = ((agent.pasos//args.checkpoint_cada)+1)*args.checkpoint_cada
    next_eval = ((agent.pasos//args.evaluar_cada)+1)*args.evaluar_cada if args.evaluar_cada else objetivo+1
    print(f'INICIO {args.algoritmo} red={args.red} GPU={agent.device} '
          f'pasos={agent.pasos}->{objetivo} pesos={folder} logs={logs}', flush=True)
    # Estos son argumentos del experimento, no un archivo de configuración.
    with (logs/'argumentos.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['argumento','valor'])
        for key, value in vars(args).items():
            writer.writerow([key, str(value)])
    with local_server(args.semilla, args.puerto) as (host, port):
        env = ConduccionEnv(host, port, args)
        env.defensor = potencia_entrenamiento(agent, args)
        obs, _ = env.reset(seed=args.semilla)
        total = 0.
        ciclos = 0
        recuperaciones = 0
        try:
            while agent.pasos < objetivo:
                action_id, params, command = agent.actuar(obs)
                new_obs, reward, done, _, info = env.step(command)
                agent.replay.add(obs, action_id, params, reward, new_obs, done)
                agent.pasos += 1
                total += args.gamma**ciclos*reward
                ciclos += 1
                recuperaciones += int(info['recuperado'])
                obs = new_obs
                if (agent.pasos > args.warmup and agent.replay.n >= args.batch
                        and agent.pasos % args.train_freq == 0):
                    for _ in range(args.gradient_steps):
                        last = agent.actualizar()
                if done:
                    agent.episodios += 1
                    csv_row(logs/'episodios.csv', dict(
                        pasos=agent.pasos, episodio=agent.episodios, retorno=env.base_return,
                        retorno_shaping=total, exito=int(info['success']), avance=info['avance'],
                        ciclos=ciclos, motivo=info['reason'], defensor=env.defensor,
                        recuperaciones=recuperaciones))
                    print(f'EPISODIO {agent.episodios} paso={agent.pasos} '
                          f'retorno={env.base_return:.3f} retorno_pbrs={total:.3f} '
                          f'exito={int(info["success"])} avance={info["avance"]:.2f}m '
                          f'motivo={info["reason"]} defensor={env.defensor} '
                          f'recuperaciones={recuperaciones}', flush=True)
                    registrar_rendimiento(agent, args, info['success'])
                    env.defensor = potencia_entrenamiento(agent, args)
                    obs, _ = env.reset()
                    total = 0.; ciclos = 0; recuperaciones = 0
                if agent.pasos % args.log_cada == 0 or agent.pasos == objetivo:
                    row = dict(pasos=agent.pasos, actualizaciones=agent.updates,
                               segundos=time.monotonic()-start, epsilon=agent.epsilon(),
                               defensor=env.defensor, **last)
                    csv_row(logs/'actualizaciones.csv', row)
                    print(f'[{args.algoritmo}/{args.corrida}] paso={agent.pasos}/{objetivo} '
                          f'updates={agent.updates} epsilon={agent.epsilon():.3f} '
                          f'loss_q={last["perdida_q"]} avance={info["avance"]:.2f}m '
                          f'episodios={agent.episodios} '
                          f'etapa={agent.curriculum["indice"]} '
                          f'exito_ventana={float(np.mean(agent.curriculum["ventana"])) if agent.curriculum["ventana"] else 0.:.1%}', flush=True)
                if agent.pasos >= next_eval or agent.pasos == objetivo:
                    result = evaluar_agente(agent, env, args, args.episodios_validacion,
                                            args.semilla_validacion,
                                            logs/'validacion_episodios.csv')
                    csv_row(logs/'validacion.csv', dict(pasos=agent.pasos, **result))
                    print(f'VALIDACION {args.algoritmo} paso={agent.pasos} '
                          f'exito={result["exito"]:.1%} retorno={result["retorno"]:.3f} '
                          f'avance={result["avance"]:.2f}m defensor={args.defensor}', flush=True)
                    score = (result['exito'], result['avance'])
                    if score > best_score:
                        best_score = score
                        agent.guardar(best_path)
                        best_data = torch.load(best_path, map_location='cpu', weights_only=True)
                        best_data['validacion'] = result
                        torch.save(best_data, best_path)
                    next_eval = agent.pasos+args.evaluar_cada if args.evaluar_cada else objetivo+1
                    # La validación reinicia el mundo. Se descarta el episodio de
                    # entrenamiento parcial, nunca se añade como fracaso o éxito.
                    env.defensor = potencia_entrenamiento(agent, args)
                    obs, _ = env.reset()
                    total = 0.; ciclos = 0; recuperaciones = 0
                if agent.pasos >= next_save:
                    agent.guardar(folder/'checkpoint.pth')
                    next_save = agent.pasos+args.checkpoint_cada
        except KeyboardInterrupt:
            agent.guardar(folder/'checkpoint.pth')
            print(f'INTERRUMPIDO: {folder}/checkpoint.pth', flush=True)
            raise SystemExit(130)
        finally:
            env.close()
    agent.guardar(folder/'final.pth')
    (folder/'checkpoint.pth').unlink(missing_ok=True)
    print(f'FINAL {folder}/final.pth ({agent.pasos} pasos)', flush=True)


def evaluar_checkpoint(args):
    from argparse import Namespace
    data = torch.load(args.pesos, map_location='cpu', weights_only=True)
    if data.get('formato') != ConduccionEnv.VERSION:
        raise ValueError('Usa un checkpoint nuevo de conducción con recuperación')
    spec = Namespace(**data['argumentos'])
    spec.dispositivo = args.dispositivo
    spec.puerto = args.puerto
    if args.defensor is not None:
        spec.defensor = args.defensor
    agent = AgenteConduccion(spec)
    agent.restaurar(data, entrenar=False)
    torch.set_num_threads(spec.hilos)
    output = Path(args.salida) if args.salida else (
        Path('resultados/logs/conduccion')/spec.algoritmo/spec.corrida/
        f'evaluacion_semilla_{args.semilla_prueba}.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'{output} existe; cambia --salida o --semilla-prueba')
    with local_server(args.semilla_prueba, spec.puerto) as (host, port):
        env = ConduccionEnv(host, port, spec)
        try:
            result = evaluar_agente(agent, env, spec, args.episodios, args.semilla_prueba, output)
        finally:
            env.close()
    print(f'EVALUACION {spec.algoritmo}: éxito={result["exito"]:.1%} '
          f'retorno={result["retorno"]:.3f} avance={result["avance"]:.2f}m '
          f'episodios={result["episodios"]} -> {output}', flush=True)
