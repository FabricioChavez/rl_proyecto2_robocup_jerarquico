"""Imitación con experto HUMANO: demostraciones, Behavioral Cloning y DAgger en PyTorch.

Flujo por tarea (persecucion, conduccion, definicion y después jerarquico):

  demos  -> juegas en tiempo real con W/A/S/D + T; se guardan TODOS los ciclos
            y se marca cuáles controlaste tú (columna humano).
  bc     -> ajuste supervisado de la MISMA red PPO/A2C/DQN, sin servidor:
              PPO/A2C: entropía cruzada  -log pi(a*|s)  + crítico a retornos Monte Carlo.
              DQN:     margen DQfD  max_a[Q(s,a)+l(a*,a)] - Q(s,a*)  + TD(0) Double-DQN.
  dagger -> la red juega; cuando pulsas teclas tomas el control (HG-DAgger,
            Kelly et al. 2019). Tus correcciones se agregan al CSV y la red se
            reajusta sobre D_1 ∪ ... ∪ D_i al terminar cada ronda.

Las teclas son de videojuego y se traducen a la acción del MENÚ DISCRETO de la tarea
(girar ±35, dash, toque, pase, tiro); esa acción es la etiqueta que se imita.

Pesos: resultados/modelo_weights/<tarea>/<algoritmo>/<corrida>/final.pth (y bc.pth),
mismo formato que entrenar, así que funcionan con evaluar, ver y entrenar --continuar.
"""

import contextlib
import csv
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from entorno.acciones import (
    BASE_CORRER,
    DIRECCIONES_CAMPO,
    LABELS,
    N_ACTIONS,
    OBJETIVOS_TIRO,
    TASKS,
    relative,
)
from modelo.arquitecturas import crear_modelo
from modelo.pesos import (
    arquitectura,
    cargar_politica,
    guardar_checkpoint,
    leer_checkpoint,
    restaurar_entrenamiento,
)
from rutas import carpeta, nombre_tarea, ruta_pesos

N_OBS = 34
OBS = [f"observacion_{i}" for i in range(N_OBS)]
META = [
    "humano",
    "accion",
    "ejecutada",
    "sugerida",
    "recompensa",
    "terminal",
    "exito",
    "gol",
    "episodio_id",
    "paso",
    "modo",
    "ronda",
    "origen",
    "tarea",
    "escenario",
    "potencia_defensor",
    "reglas",
]
COLUMNAS = OBS + META

# --------------------------------------------------------------------------- teclas

AL_PIE = 1.0  # metros: el balón se puede patear (kickable ~1.085 m)
# dash, giro +35, giro -35 en el menú de cada tarea
MOVIMIENTO = {"persecucion": (0, 2, 3), "conduccion": (0, 1, 2), "definicion": (5, 6, 7)}
MIRA_MAX, MIRA_PASO = 6.5, 0.5  # metros sobre la línea de gol; 7/8 la mueven
AYUDA = {
    "persecucion": "W A S D / flechas: moverte (W+D = diagonal). El jugador gira y corre solo.",
    "conduccion": (
        "W A S D: moverte. Con el balón al pie, moverte lo conduce (D al arco, W arriba, S abajo).\n"
        "7: robar = patear el balón hacia el arco cuando está a tu alcance (1 m), aunque lo lleve el rival.   8 (o E): pase."
    ),
    "definicion": (
        "W A S D: acomodarte.   ESPACIO: tirar de frente (hacia donde miras).   7 / 8: subir / bajar la mira para apuntar.\n"
        "Apunta al lado del arco que el arquero deja libre."
    ),
    "global": "1 persecución   2 conducción/pase   3 definición   (la habilidad sigue hasta que cambies)",
}
AYUDA_COMUN = "\nX: repetir episodio (lo descarta)   Q / Esc: guardar y terminar.   Esta ventana debe tener el foco."


def direccion(teclas):
    """Ángulo en coordenadas del campo (y crece hacia ABAJO en el monitor)."""
    dx = ("d" in teclas) - ("a" in teclas)
    dy = ("s" in teclas) - ("w" in teclas)
    return None if dx == 0 and dy == 0 else math.degrees(math.atan2(dy, dx))


def girar_o_correr(env, task, angulo):
    p = env.world.players[("Attack", 1)]
    diferencia = (angulo - p[4] + 180) % 360 - 180
    dash, giro_mas, giro_menos = MOVIMIENTO[task]
    if abs(diferencia) <= 17.5:
        return dash  # de frente: corre a potencia completa
    if math.hypot(p[2], p[3]) < 0.35:
        # Parado, girar es rápido (35 grados por ciclo).
        return giro_mas if diferencia > 0 else giro_menos
    # Corriendo, girar es lento (~12 grados por ciclo): corre de lado o hacia
    # atrás en la dirección del campo que pides, sin girar el cuerpo.
    cercana = min(
        range(len(DIRECCIONES_CAMPO)),
        key=lambda i: abs((DIRECCIONES_CAMPO[i] - angulo + 180) % 360 - 180),
    )
    return BASE_CORRER[task] + cercana


def quite_posible(env):
    """Reglas 'recuperacion': balón a tu alcance de patada (SoccerEnv.robo_posible)."""
    return env.robo_posible()


def mira_de_frente(env):
    """Punto de la línea de gol hacia el que mira el cuerpo del jugador.
    Si no mira hacia el arco, se usa su propia altura (tiro recto al frente)."""
    p = env.world.players[("Attack", 1)]
    rumbo = math.radians(p[4])
    y = p[1] if math.cos(rumbo) < 0.1 else p[1] + math.tan(rumbo) * (52.5 - p[0])
    return float(np.clip(y, -MIRA_MAX, MIRA_MAX))


def tiro_mas_cercano(mira):
    return min(OBJETIVOS_TIRO, key=lambda index: abs(OBJETIVOS_TIRO[index] - mira))


def accion_humana(teclas, env, task, mira=None):
    """Teclas mantenidas -> índice del menú discreto, o None si no estás actuando."""
    angulo = direccion(teclas)
    al_pie = relative(env.world.players[("Attack", 1)], env.world.ball)[0] <= AL_PIE
    if task == "conduccion":
        # Solo se etiqueta el robo cuando funciona; si no, sigues moviéndote.
        if "7" in teclas and quite_posible(env):
            return 7  # quitar / robar (tackle)
        if "8" in teclas or "e" in teclas:
            return 6  # pase
        if al_pie and angulo is not None and abs(angulo) < 150:
            if angulo <= -30:
                return 4  # drible arriba
            if angulo >= 30:
                return 5  # drible abajo
            return 3  # drible al arco
    if task == "definicion" and "espacio" in teclas and al_pie:
        return tiro_mas_cercano(mira_de_frente(env) if mira is None else mira)
    if angulo is None:
        return None
    return girar_o_correr(env, task, angulo)


class Espacios:
    """Solo espacios Gymnasium: BC no necesita servidor."""

    def __init__(self, task):
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (N_OBS,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS[task])


# --------------------------------------------------------------------------- datos


def cargar_datos(path, task):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNAS)
    data = pd.read_csv(path)
    if list(data.columns) != COLUMNAS:
        antiguo = path.with_name(path.stem + f"_formato_anterior_{datetime.now():%H%M%S}.csv")
        path.rename(antiguo)
        print(f"  {path} tenía el formato anterior; se apartó como {antiguo}", flush=True)
        return pd.DataFrame(columns=COLUMNAS)
    if len(data) and set(data.tarea.astype(str)) != {nombre_tarea(task)}:
        raise ValueError(f"{path} contiene demostraciones de otra tarea")
    humanas = data[data.humano == 1]
    if len(humanas) and (humanas.accion.min() < 0 or humanas.accion.max() >= N_ACTIONS[task]):
        raise ValueError(f"{path} contiene acciones fuera del rango de {task}")
    return data


def agregar_filas(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows, columns=COLUMNAS).to_csv(path, mode="a", header=nuevo, index=False)


def agregar_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists() or path.stat().st_size == 0
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not nuevo:
        with path.open(encoding="utf-8", newline="") as stream:
            previous = next(csv.reader(stream), [])
        if previous != fields:
            old = pd.read_csv(path)
            pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
            return
    pd.DataFrame(rows, columns=fields).to_csv(path, mode="a", header=nuevo, index=False)


def tensores(data, gamma, device):
    """Etiquetas humanas, retornos Monte Carlo y transiciones TD, por episodio."""
    data = data.reset_index(drop=True)
    n = len(data)
    x = data[OBS].to_numpy(np.float32)
    humano = data.humano.to_numpy(np.int64) == 1
    etiqueta = np.where(humano, data.accion.to_numpy(np.int64), 0)
    ejecutada = data.ejecutada.to_numpy(np.int64)
    reward = data.recompensa.fillna(0).to_numpy(np.float32)
    terminal = data.terminal.fillna(0).to_numpy(np.float32)
    episodio = data.episodio_id.astype(str).to_numpy()
    paso = data.paso.to_numpy(np.int64)
    sigue = np.zeros(n, dtype=bool)
    sigue[:-1] = (episodio[1:] == episodio[:-1]) & (paso[1:] == paso[:-1] + 1)
    x_next = np.roll(x, -1, axis=0)
    # TD solo con acción del menú y siguiente estado consecutivo (o estado terminal).
    td_valido = (ejecutada >= 0) & (sigue | (terminal > 0))
    retorno = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in range(n - 1, -1, -1):
        if not sigue[i]:
            running = 0.0
        running = reward[i] + gamma * running
        retorno[i] = running

    def t(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return dict(
        x=t(x),
        humano=t(humano, torch.bool),
        y=t(etiqueta, torch.long),
        ejecutada=t(np.maximum(ejecutada, 0), torch.long),
        r=t(reward),
        done=t(terminal),
        x_next=t(x_next),
        td=t(td_valido, torch.bool),
        retorno=t(retorno),
    )


# --------------------------------------------------------------------------- modelo


def preparar_modelo(args, task):
    checkpoint = leer_checkpoint(args.resume) if args.resume else None
    if checkpoint:
        if checkpoint.get("generador") != "torch_propio":
            raise ValueError("BC/DAgger solo continúa pesos PyTorch propios")
        if checkpoint["algoritmo"] != args.algo or checkpoint["tarea"] != task:
            raise ValueError(
                f"{args.resume} es {checkpoint['tarea']}/{checkpoint['algoritmo']}, "
                f"no {task}/{args.algo}"
            )
        inferred = arquitectura(checkpoint)[2]
        if args.network and args.network != inferred:
            raise ValueError("--red no coincide con los pesos")
        args.network = inferred
    else:
        args.network = args.network or [128, 128]
    model = crear_modelo(Espacios(task), args)
    if checkpoint:
        restaurar_entrenamiento(model, checkpoint)
    return model


@torch.no_grad()
def precision(model, x, y, dqn):
    if len(x) == 0:
        return float("nan")
    hits = 0
    for xb, yb in zip(x.split(4096), y.split(4096)):
        scores = model.policy.q_net(xb) if dqn else model.policy(xb)[0].logits
        hits += int((scores.argmax(-1) == yb).sum())
    return hits / len(x)


def entrenar_bc(model, data, args, ronda):
    """Behavioral Cloning sobre TODO el dataset agregado (DAgger = BC sobre D_1 ∪ ... ∪ D_i)."""
    dqn = args.algo == "dqn"
    device = model.device
    batch = tensores(data, model.gamma, device)
    n = len(batch["x"])
    n_humanas = int(batch["humano"].sum())
    if n_humanas == 0:
        raise ValueError("El dataset no tiene ciclos controlados por ti")
    generator = torch.Generator().manual_seed(args.seed * 1000 + ronda)
    perm = torch.randperm(n, generator=generator).to(device)
    n_val = int(round(n * args.validation)) if n_humanas >= 50 else 0
    val, train = perm[:n_val], perm[n_val:]
    train_h = train[batch["humano"][train]]
    val_h = val[batch["humano"][val]]
    actions = N_ACTIONS[args.task]
    weight = None
    if args.balance:
        counts = torch.bincount(batch["y"][train_h], minlength=actions).clamp(min=1)
        weight = (len(train_h) / (actions * counts.float())).clamp(max=10.0)
    optimizer = model.policy.optimizer
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    if dqn:
        grupos = [list(model.policy.q_net.parameters())]
    else:
        extractor = model.policy.mlp_extractor
        grupos = [
            [*extractor.policy_net.parameters(), *model.policy.action_net.parameters()],
            [*extractor.value_net.parameters(), *model.policy.value_net.parameters()],
        ]
    clip = args.max_grad_norm or (10.0 if dqn else 0.5)
    gamma = model.gamma
    zero = torch.zeros((), device=device)
    rows = []
    every = max(1, args.bc_epochs // 10)
    print(
        f"  {n} ciclos ({n_humanas} controlados por ti), validación {len(val_h)} etiquetas",
        flush=True,
    )
    for epoch in range(1, args.bc_epochs + 1):
        model.policy.set_training_mode(True)
        if dqn:
            model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
        stats = defaultdict(list)
        order = train[torch.randperm(len(train), generator=generator).to(device)]
        for idx in order.split(args.batch):
            h = idx[batch["humano"][idx]]
            if dqn:
                imitation = zero
                if len(h):
                    q_h = model.policy.q_net(batch["x"][h])
                    y_h = batch["y"][h]
                    margin = torch.full_like(q_h, args.margin)
                    margin.scatter_(1, y_h[:, None], 0.0)
                    large_margin = (q_h + margin).max(1).values - q_h.gather(1, y_h[:, None]).squeeze(1)
                    if weight is not None:
                        large_margin = large_margin * weight[y_h]
                    imitation = large_margin.mean()
                auxiliary = zero
                valid = idx[batch["td"][idx]]
                if args.td_weight > 0 and len(valid):
                    with torch.no_grad():
                        nxt = batch["x_next"][valid]
                        best = model.policy.q_net(nxt).argmax(1, keepdim=True)
                        future = model.policy.q_net_target(nxt).gather(1, best).squeeze(1)
                        target = batch["r"][valid] + gamma * (1 - batch["done"][valid]) * future
                    taken = model.policy.q_net(batch["x"][valid]).gather(
                        1, batch["ejecutada"][valid][:, None]
                    ).squeeze(1)
                    auxiliary = F.smooth_l1_loss(taken, target)
                loss = imitation + args.td_weight * auxiliary
            else:
                distribution, value = model.policy(batch["x"][idx])
                auxiliary = F.mse_loss(value, batch["retorno"][idx])
                imitation = zero
                if len(h):
                    mask = batch["humano"][idx]
                    logits = distribution.logits[mask]
                    imitation = F.cross_entropy(logits, batch["y"][h], weight=weight)
                    imitation = imitation - args.bc_entropy * distribution.entropy()[mask].mean()
                loss = imitation + args.bc_value_coef * auxiliary
            if not loss.requires_grad:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for params in grupos:
                torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            model._n_updates += 1
            stats["imitacion"].append(float(imitation))
            stats["auxiliar"].append(float(auxiliary))
        model.policy.set_training_mode(False)
        row = dict(
            ronda=ronda,
            epoca=epoch,
            ciclos=n,
            etiquetas_humanas=n_humanas,
            etiquetas_validacion=len(val_h),
            perdida_imitacion=float(np.mean(stats["imitacion"] or [np.nan])),
            perdida_auxiliar=float(np.mean(stats["auxiliar"] or [np.nan])),
            precision_entrenamiento=precision(model, batch["x"][train_h], batch["y"][train_h], dqn),
            precision_validacion=precision(model, batch["x"][val_h], batch["y"][val_h], dqn),
            fecha=datetime.now().isoformat(timespec="seconds"),
        )
        rows.append(row)
        if epoch % every == 0 or epoch == args.bc_epochs:
            print(
                f"  BC ronda {ronda} época {epoch}/{args.bc_epochs}: "
                f"imitación={row['perdida_imitacion']:.4f} aux={row['perdida_auxiliar']:.4f} "
                f"acierto entren.={row['precision_entrenamiento']:.1%} "
                f"valid.={row['precision_validacion']:.1%}",
                flush=True,
            )
    if dqn:
        model.policy.q_net_target.load_state_dict(model.policy.q_net.state_dict())
    model.policy.set_training_mode(False)
    return rows


# --------------------------------------------------------------------------- juego


def estado_texto(env, task, ep, total, extra="", mira=None):
    w = env.world
    p = w.players[("Attack", 1)]
    distance, angle = relative(p, w.ball)
    defender = relative(p, w.players[("Rival", 2)])[0]
    linea = ""
    if task == "definicion":
        origen = "de frente" if mira is None else "con 7/8"
        mira = mira_de_frente(env) if mira is None else mira
        linea = (
            f"MIRA {origen} y={mira:+4.1f} m ({LABELS[task][tiro_mas_cercano(mira)]})   "
            f"arquero en y={w.players[('Rival', 1)][1]:+4.1f}\n"
        )
    elif getattr(env, "posesion_rival", 0):
        linea = (
            f"¡el rival tiene el balón! {env.posesion_rival}/{env.ciclos_posesion} ciclos   "
            + ("ROBO POSIBLE: pulsa 7" if quite_posible(env) else "pégate al balón (1 m)")
            + "\n"
        )
    return (
        f"{nombre_tarea(task)}   episodio {ep}/{total}   ciclo {env.t}/{env.horizon}\n"
        f"balón a {distance:4.1f} m ({angle:+4.0f}°)   x del balón {w.ball[0]:+5.1f}\n"
        f"defensor a {defender:4.1f} m   estamina {env.server.stamina:5.0f}\n{linea}{extra}"
    )


def fila(observation, **values):
    row = dict(zip(OBS, observation.astype(float).tolist()))
    row.update(values)
    return row


def episodio(env, task, model, skills, control, args, modo, ronda, semilla, ep, total):
    """Un episodio en tiempo real. Devuelve (filas, salir, descartar, info)."""
    env.reset(seed=semilla)
    # En modo síncrono rcssserver solo dibuja al avanzar: dos ciclos sin acción
    # muestran la posición inicial. No son pasos del MDP.
    for _ in range(2):
        env.server.tick(["(turn 0)"] * 4)
    env.last_ball = env.world.ball[:2].copy()
    validas = {"w", "a", "s", "d", "t", "e", "r", "i", "k", "espacio", "1", "2", "3", "7", "8"}
    mira = None  # None = de frente; 7/8 la fijan y la mueven
    texto = estado_texto(
        env, task, ep, total, "Pulsa una tecla de juego (o espacio) para empezar.", mira
    )
    pausa = contextlib.nullcontext()
    if not args.no_pause:
        from entorno.pausa import pause_managed_server

        pausa = pause_managed_server(args.port)
    with pausa:
        inicio = control.esperar(texto, validas)
    if inicio is None:
        return [], True, False, {}
    ep_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{modo}_r{ronda}"
    rows, info = [], {}
    opcion = int(inicio) - 1 if inicio in ("1", "2", "3") else None
    humanos = intervenciones = 0
    antes = False
    # Si empezaste con ESPACIO, no dispara hasta que lo sueltes.
    soltar_espacio = inicio == "espacio"
    while True:
        started = time.monotonic()
        teclas, eventos = control.leer()
        if soltar_espacio:
            if "espacio" in teclas and "espacio" not in eventos:
                teclas = teclas - {"espacio"}
            else:
                soltar_espacio = False
        if "q" in eventos or "esc" in eventos:
            return rows, True, False, info
        if "x" in eventos:
            return [], False, True, info
        if task == "definicion" and ("7" in teclas or "8" in teclas):
            if mira is None:
                mira = mira_de_frente(env)  # empieza donde miras
            if "7" in teclas:
                mira = max(-MIRA_MAX, mira - MIRA_PASO)
            if "8" in teclas:
                mira = min(MIRA_MAX, mira + MIRA_PASO)
        observation = env.observe()
        sugerida = int(model.predict(observation, deterministic=True)[0]) if model else None
        if task == "global":
            nuevas = [int(k) - 1 for k in eventos if k in ("1", "2", "3")]
            if nuevas:
                opcion = nuevas[-1]
            # demos: tu habilidad elegida sigue activa; dagger: solo cuentas al pulsar.
            humana = opcion if (modo == "demos" or nuevas) else None
            if humana is None and model is None:
                control.estado("ELIGE 1 / 2 / 3", "#8a8980", estado_texto(env, task, ep, total))
                time.sleep(0.03)
                continue
        else:
            humana = accion_humana(teclas, env, task, mira)
        ejecutada = humana if humana is not None else sugerida
        if task == "global":
            reward, terminated, truncated, info = ejecutar_opcion(
                env, skills, ejecutada, args.fps, control, task, ep, total
            )
        elif ejecutada is None:
            # demos sin teclas: el jugador no hace nada y el ciclo no se etiqueta.
            _, reward, terminated, truncated, info = env._cycle(0, task, command_override="(turn 0)")
        else:
            robo = task == "conduccion" and ejecutada == 7
            if robo:
                atacante = env.world.players[("Attack", 1)]
                previo = relative(atacante, env.world.ball)[0], env.world.ball[:2].copy()
            _, reward, terminated, truncated, info = env.step(ejecutada)
            if robo:
                movido = float(np.linalg.norm(env.world.ball[:2] - previo[1]))
                print(
                    f"  [7] patada de robo con el balón a {previo[0]:.2f} m: "
                    f"el balón se movió {movido:.2f} m",
                    flush=True,
                )
        es_humana = humana is not None
        humanos += int(es_humana)
        intervenciones += int(es_humana and not antes)
        antes = es_humana
        rows.append(
            fila(
                observation,
                humano=int(es_humana),
                accion=humana if es_humana else -1,
                ejecutada=-1 if ejecutada is None else ejecutada,
                sugerida=-1 if sugerida is None else sugerida,
                recompensa=float(reward),
                terminal=int(terminated or truncated),
                exito=int(info.get("success", False)),
                gol=int(info.get("goal", False)),
                episodio_id=ep_id,
                paso=len(rows),
                modo=modo,
                ronda=ronda,
                origen="humano",
                tarea=nombre_tarea(task),
                escenario=args.scenario,
                potencia_defensor=args.defender_power,
                reglas=env.reglas,
            )
        )
        etiqueta = LABELS[task][ejecutada] if ejecutada is not None else "nada"
        if es_humana:
            modo_texto, color = "TÚ", "#d03b3b"
        elif sugerida is not None:
            modo_texto, color = "RED", "#0ca30c"
        else:
            modo_texto, color = "ESPERANDO TECLAS", "#8a8980"
        control.estado(
            modo_texto,
            color,
            estado_texto(
                env, task, ep, total,
                f"acción: {etiqueta}   ciclos tuyos {humanos}   intervenciones {intervenciones}",
                mira,
            ),
        )
        if terminated or truncated:
            break
        if task != "global" and args.fps:
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - started)))
    resultado = "ÉXITO" if info.get("success") else ("tiempo agotado" if info.get("timeout") else "fallo")
    print(
        f"Episodio {ep}/{total}: {resultado}{' (gol)' if info.get('goal') else ''}, "
        f"{env.t} ciclos, {humanos} tuyos, {intervenciones} intervenciones",
        flush=True,
    )
    return rows, False, False, info


# --------------------------------------------------------------------------- experto programado


def experto_conduccion(env):
    """Experto automático de conducción. Toma del defensor azul su regla simple
    (ir derecho al balón y tocarlo apenas está al alcance) y la orienta al objetivo
    del atacante: superar la línea del defensor."""
    w = env.world
    p, d = w.players[("Attack", 1)], w.players[("Rival", 2)]
    mate = w.players[("Attack", 2)]
    dist, ang = relative(p, w.ball)
    if env.robo_posible() and relative(d, w.ball)[0] < dist + 0.3:
        return 7  # disputa: el rival lo tiene y está a tu alcance
    if dist > AL_PIE:
        return (1 if ang > 0 else 2) if abs(ang) > 17.5 else 0  # ir al balón
    dx, dy = d[0] - p[0], d[1] - p[1]
    libre = relative(d, mate)[0] > 4 and mate[0] > d[0]
    if env.scenario == "2v1" and 0 < dx < 6 and abs(dy) < 3 and libre:
        return 6  # pase al compañero que ya está pasada la línea
    if 0 < dx < 10 and abs(dy) < 4:
        return 4 if dy > 0 else 5  # esquivar: drible diagonal al lado libre
    return 3  # camino libre: drible al arco


def experto_global(env, radio=5.0):
    """Experto programado del coordinador (opción 0/1/2). Persecución con el balón suelto
    lejos; definición con el balón al pie en la zona de tiro (x > X_DEFINICION del
    entorno, la misma beta(s) que termina conducción); si no, conducción."""
    p = env.world.players[("Attack", 1)]
    d = relative(p, env.world.ball)[0]
    if d > radio:
        return 0
    return 2 if (d <= env.RADIO_TIRO and p[0] > env.X_DEFINICION) else 1


EXPERTOS = {"conduccion": experto_conduccion, "global": experto_global}


def episodio_programado(env, task, model, args, modo, ronda, semilla, ep, total):
    """demos: actúa el experto. dagger (Ross et al. 2011): actúa la red y el experto
    etiqueta TODOS los estados que visita. Sin ventana ni tiempo real."""
    env.reset(seed=semilla)
    ep_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{modo}_programado_r{ronda}"
    rows, info, acuerdos = [], {}, []
    while True:
        started = time.monotonic()
        observation = env.observe()
        experta = EXPERTOS[task](env)
        sugerida = int(model.predict(observation, deterministic=True)[0]) if model else None
        ejecutada = experta if sugerida is None else sugerida
        _, reward, terminated, truncated, info = env.step(ejecutada)
        if sugerida is not None:
            acuerdos.append(sugerida == experta)
        rows.append(
            fila(
                observation,
                humano=1,  # = estado etiquetado por el experto
                accion=experta,
                ejecutada=ejecutada,
                sugerida=-1 if sugerida is None else sugerida,
                recompensa=float(reward),
                terminal=int(terminated or truncated),
                exito=int(info.get("success", False)),
                gol=int(info.get("goal", False)),
                episodio_id=ep_id,
                paso=len(rows),
                modo=modo,
                ronda=ronda,
                origen="programado",
                tarea=nombre_tarea(task),
                escenario=args.scenario,
                potencia_defensor=args.defender_power,
                reglas=env.reglas,
            )
        )
        if terminated or truncated:
            break
        if not args.no_monitor and args.fps:  # con monitor, a velocidad visible
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - started)))
    resultado = "ÉXITO" if info.get("success") else ("tiempo agotado" if info.get("timeout") else "fallo")
    acuerdo = f", acuerdo red-experto {np.mean(acuerdos):.0%}" if acuerdos else ""
    print(f"Episodio {ep}/{total}: {resultado}, {env.t} ciclos{acuerdo}", flush=True)
    return rows, False, False, info


def ejecutar_opcion(env, skills, option, fps, control, task, ep, total):
    """Mismo intervalo fijo que SoccerEnv.step('global'), ciclo a ciclo para verlo."""
    total_reward = 0.0
    for j in range(env.option_steps):
        started = time.monotonic()
        low = int(skills[option].predict(env.observe(), deterministic=True)[0])
        _, reward, terminated, truncated, info = env._cycle(low, TASKS[option])
        total_reward += env.gamma**j * reward
        control.refrescar()  # mantiene viva la ventana; tus pulsaciones se leen al decidir
        if fps:
            time.sleep(max(0.0, 1 / fps - (time.monotonic() - started)))
        if terminated or truncated:
            break
    info.update(option=TASKS[option], primitive_steps=j + 1)
    return total_reward, terminated, truncated, info


# --------------------------------------------------------------------------- comando


def validar(args):
    if not 0 <= args.validation < 0.5:
        raise ValueError("--validacion debe estar en [0, 0.5)")
    if not 0 <= args.defender_power <= 100:
        raise ValueError("--fuerza-defensor debe estar entre 0 y 100")
    for name in ("episodes", "rounds", "bc_epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} debe ser positivo")
    if args.fps <= 0:
        raise ValueError("--fps debe ser positivo")
    if args.run and args.run.startswith("sb3_"):
        raise ValueError("sb3_ identifica referencias históricas; elige otra --corrida")


def guardar(model, weights, nombres, task, algo):
    weights.mkdir(parents=True, exist_ok=True)
    for name in nombres:
        guardar_checkpoint(model, weights / name, task, algo)
    print(f"  Pesos guardados: {', '.join(str(weights / n) for n in nombres)}", flush=True)


def ejecutar(args):
    validar(args)
    args.reglas = args.reglas or "recuperacion"
    task = args.task
    programado = getattr(args, "expert", "humano") == "programado"
    if programado and task not in EXPERTOS:
        raise ValueError(f"--experto programado solo existe para: {', '.join(EXPERTOS)}")
    origen = "experto" if programado else "humano"
    args.run = args.run or origen
    weights = carpeta("modelo_weights", task, args.algo, args.seed, args.run)
    logs = carpeta("logs", task, args.algo, args.seed, args.run)
    logs.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset or Path("datos/demostraciones") / f"{origen}_{nombre_tarea(task)}.csv"
    data = cargar_datos(dataset, task)

    if args.mode == "bc":
        if data.empty:
            raise ValueError(
                f"{dataset} no tiene demostraciones. Primero: python proyecto.py imitar demos "
                f"--tarea {nombre_tarea(task)}"
            )
        reglas = set(data.reglas.astype(str))
        if len(reglas) != 1:
            raise ValueError(f"{dataset} mezcla reglas {reglas}; separa los CSV con --datos")
        model = preparar_modelo(args, task)
        model.reglas = reglas.pop()
        print(
            f"BC {nombre_tarea(task)}/{args.algo} (reglas {model.reglas}): {data.episodio_id.nunique()} episodios humanos, "
            f"dispositivo {model.device}",
            flush=True,
        )
        agregar_csv(logs / "imitacion.csv", entrenar_bc(model, data, args, 0))
        guardar(model, weights, ["bc.pth", "final.pth"], task, args.algo)
        return

    model = None
    if args.mode == "dagger":
        args.resume = args.resume or weights / "final.pth"
        if not Path(args.resume).exists():
            raise FileNotFoundError(
                f"{args.resume} no existe. Primero: python proyecto.py imitar bc "
                f"--tarea {nombre_tarea(task)} --algoritmo {args.algo} --corrida {args.run}"
            )
        model = preparar_modelo(args, task)
        model.reglas = args.reglas
    skills = None
    if task == "global":
        skill_algo = args.skill_algo or args.algo
        skill_run = args.skills_run or args.run
        paths = [ruta_pesos(t, skill_algo, args.seed, corrida=skill_run) for t in TASKS]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Faltan las habilidades para coordinar: " + ", ".join(missing))
        skills = [cargar_politica(None, p, args.device) for p in paths]
        distintas = {s.checkpoint.get("reglas", "clasicas") for s in skills} - {args.reglas}
        if distintas:
            print(f"  Aviso: habilidades entrenadas con reglas {distintas}, juego con {args.reglas}")
        print(f"Habilidades: {', '.join(map(str, paths))}", flush=True)

    from entorno.escenarios import InteractiveEnv
    from entorno.servidor import local_server
    from entorno.teclado import Control
    from evaluador.visualizar import official_monitor

    contador = int(data.episodio_id.nunique()) if len(data) else 0
    rondas = list(range(1, args.rounds + 1)) if args.mode == "dagger" else [0]
    titulo = f"Experto {origen} - {nombre_tarea(task)} ({args.mode})"
    with local_server(args.seed, args.port) as (host, port):
        env = InteractiveEnv(
            task=task,
            host=host,
            port=port,
            scenario=args.scenario,
            skills=skills,
            shaping=args.shaping,
            gamma=args.gamma,
            horizon=args.horizon,
            option_steps=args.option_steps,
            reglas=args.reglas,
            ciclos_posesion=args.possession_cycles,
        )
        env.conditions = dict(
            defensor="perseguir", potencia_defensor=args.defender_power, arquero_activo=True
        )
        monitor = contextlib.nullcontext() if args.no_monitor else official_monitor(port, logs / "monitor")
        try:
            ventana = (
                contextlib.nullcontext() if programado
                else Control(titulo, AYUDA[task] + AYUDA_COMUN)
            )
            with monitor, ventana as control:
                print(f"{titulo}. Demostraciones -> {dataset}", flush=True)
                salir = False
                for ronda in rondas:
                    if args.mode == "dagger":
                        print(
                            f"=== DAgger ronda {ronda}/{args.rounds}: la red juega; "
                            + ("el experto etiqueta cada estado ===" if programado
                               else "pulsa teclas para corregirla ==="),
                            flush=True,
                        )
                    resumen = []
                    ep = 0
                    while ep < args.episodes:
                        semilla = 50000 + 1000 * args.seed + contador
                        if programado:
                            rows, salir, descartar, info = episodio_programado(
                                env, task, model, args, args.mode, ronda,
                                semilla, ep + 1, args.episodes,
                            )
                        else:
                            rows, salir, descartar, info = episodio(
                                env, task, model, skills, control, args, args.mode, ronda,
                                semilla, ep + 1, args.episodes,
                            )
                        contador += 1
                        if descartar:
                            print("Episodio descartado; se repite.", flush=True)
                            continue
                        agregar_filas(dataset, rows)
                        if rows:
                            humanas = [r for r in rows if r["humano"]]
                            resumen.append(
                                dict(
                                    modo=args.mode,
                                    ronda=ronda,
                                    episodio_id=rows[0]["episodio_id"],
                                    completo=int(not salir),
                                    exito=int(info.get("success", False)) if not salir else 0,
                                    gol=int(info.get("goal", False)) if not salir else 0,
                                    ciclos=len(rows),
                                    ciclos_humanos=len(humanas),
                                    acuerdo_red=(
                                        float(np.mean([r["accion"] == r["sugerida"] for r in humanas]))
                                        if model and humanas
                                        else float("nan")
                                    ),
                                    potencia_defensor=args.defender_power,
                                    escenario=args.scenario,
                                    fecha=datetime.now().isoformat(timespec="seconds"),
                                )
                            )
                        if salir:
                            break
                        ep += 1
                    agregar_csv(logs / "demostraciones.csv", resumen)
                    completos = [r for r in resumen if r["completo"]]
                    if completos:
                        print(
                            f"Ronda {ronda}: éxito {sum(r['exito'] for r in completos)}/{len(completos)}",
                            flush=True,
                        )
                    if args.mode == "dagger" and resumen:
                        data = cargar_datos(dataset, task)
                        print(f"Reajustando con el dataset agregado ({len(data)} ciclos)", flush=True)
                        agregar_csv(logs / "imitacion.csv", entrenar_bc(model, data, args, ronda))
                        guardar(model, weights, ["final.pth"], task, args.algo)
                    if salir:
                        break
        except KeyboardInterrupt:
            print("\nInterrumpido: los episodios terminados ya están en el CSV.")
        finally:
            env.close()
    if args.mode == "demos":
        total = cargar_datos(dataset, task)
        if len(total):
            print(
                f"Dataset {dataset}: {int(total.humano.sum())} ciclos etiquetados en "
                f"{total.episodio_id.nunique()} episodios. Siguiente: python proyecto.py imitar bc "
                f"--tarea {nombre_tarea(task)} --algoritmo {args.algo}"
                + (" --experto programado" if programado else ""),
                flush=True,
            )
