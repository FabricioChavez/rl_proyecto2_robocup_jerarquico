"""Ejecuta políticas P2 en el monitor OFICIAL rcssmonitor, sin panel propio."""

import contextlib
from datetime import datetime
import csv
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

from modelo.pesos import cargar_politica
from rutas import ruta_pesos

from entorno.acciones import TASKS, LABELS
from entorno.servidor import local_server
from entorno.pausa import pause_managed_server


from entorno.escenarios import InteractiveEnv, apply_scene


@contextlib.contextmanager
def official_monitor(port, output, offscreen=False, replay=None):
    windows = platform.system() == "Windows"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    distro = os.getenv("ROBOCUP_WSL_DISTRO", "Ubuntu")
    command = ["rcssmonitor"]
    if replay is None:
        command += [
            "--connect",
            "--server-host",
            "127.0.0.1",
            "--server-port",
            str(port),
            "--geometry",
            "1200x800",
            "--show-stamina",
            "on",
            "--show-player-number",
            "on",
            "--show-status-bar",
            "on",
        ]
    else:
        path = Path(replay).resolve().as_posix()
        if windows:
            path = subprocess.check_output(
                [
                    "wsl",
                    "-d",
                    os.getenv("ROBOCUP_WSL_DISTRO", "Ubuntu"),
                    "--",
                    "wslpath",
                    "-a",
                    path,
                ],
                text=True,
            ).strip()
        command.append(path)
    if windows:
        pid_path = f"/tmp/robocup-monitor-{uuid.uuid4().hex}.pid"
        if offscreen:
            command = ["env", "QT_QPA_PLATFORM=offscreen", *command]
        command = [
            "wsl",
            "-d",
            distro,
            "--exec",
            "sh",
            "-c",
            'echo $$ > "$1"; shift; exec "$@"',
            "monitor-launcher",
            pid_path,
            *command,
        ]
    elif offscreen:
        command = ["env", "QT_QPA_PLATFORM=offscreen", *command]
    with (output / "monitor.log").open("ab") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if windows else {}),
        )
        try:
            time.sleep(0.8)
            if process.poll() is not None:
                raise RuntimeError(
                    f'rcssmonitor terminó. Consulta {output / "monitor.log"}'
                )
            yield process
        finally:
            if process.poll() is None:
                if windows:
                    subprocess.run(
                        [
                            "wsl",
                            "-d",
                            distro,
                            "--exec",
                            "sh",
                            "-c",
                            'if [ -r "$1" ]; then read -r pid < "$1"; kill -TERM "$pid" 2>/dev/null || true; fi',
                            "monitor-cleanup",
                            pid_path,
                        ],
                        capture_output=True,
                        timeout=10,
                    )
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def keyboard_command(enabled):
    if not enabled:
        return None
    if platform.system() == "Windows":
        import msvcrt

        return msvcrt.getwch().lower() if msvcrt.kbhit() else None
    import select, sys

    return (
        sys.stdin.readline().strip().lower()
        if select.select([sys.stdin], [], [], 0)[0]
        else None
    )


def ejecutar(args):
    if not 1 <= args.fps <= 30 or args.episodes < 0:
        raise ValueError("fps entre 1 y 30; episodes >=0")
    output = args.output or Path(
        "resultados/logs/visualizacion"
    ) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    if args.replay:
        with official_monitor(args.port, output, args.offscreen, args.replay) as gui:
            gui.wait()
        return
    model = cargar_politica(
        args.algo,
        args.weights or ruta_pesos(args.task, args.algo, args.seed, corrida=args.run),
        args.device,
    )
    skills = (
        [
            cargar_politica(
                args.algo,
                ruta_pesos(t, args.algo, args.seed, corrida=args.run),
                args.device,
            )
            for t in TASKS
        ]
        if args.task == "global"
        else None
    )
    positions = {
        name: getattr(args, name)
        for name in ("balon", "atacante", "companero", "defensor", "arquero")
        if getattr(args, name) is not None
    }
    scene = {
        "inicial": {
            "posiciones": positions,
            "velocidad_balon": args.ball_velocity,
            "condiciones": {
                "defensor": args.defender_behavior,
                "potencia_defensor": args.defender_power,
                "arquero_activo": not args.static_goalkeeper,
            },
        }
    }
    if not args.no_keyboard:
        print(
            "Monitor oficial. Teclas en ESTA terminal: p pausa/reanuda, n un ciclo estando en pausa, r reinicia, q sale.",
            flush=True,
        )
    else:
        print(
            "Monitor oficial. Controles del árbitro disponibles en la ventana; termina la sesión cerrando el monitor.",
            flush=True,
        )
    print(
        "En el monitor: clic derecho -> Drop Ball / modos del árbitro. Menú Referee para PlayOn.",
        flush=True,
    )
    print(f"Logs: {output}. Grabación RCG: {args.record}", flush=True)
    with local_server(
        args.seed, args.port, game_log_dir=output if args.record else None
    ) as (host, port):
        env = InteractiveEnv(
            task=args.task,
            host=host,
            port=port,
            scenario=args.scenario,
            skills=skills,
            shaping="none",
            gamma=args.gamma,
            horizon=args.horizon,
            option_steps=args.option_steps,
        )
        env.conditions = {}
        try:
            with (
                official_monitor(args.port, output, args.offscreen) as gui,
                (output / "trayectoria.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as log,
            ):
                writer = csv.DictWriter(
                    log,
                    fieldnames=[
                        "episodio",
                        "paso",
                        "habilidad",
                        "accion",
                        "balon_x",
                        "balon_y",
                    ],
                )
                writer.writeheader()
                episode = 0
                paused = False
                quit_requested = False
                while gui.poll() is None and (
                    args.episodes == 0 or episode < args.episodes
                ):
                    env.conditions = {}
                    env.reset(seed=args.seed + episode)
                    apply_scene(env, scene.get("inicial", {}))
                    option = 0
                    remaining = 0
                    episode += 1
                    print(
                        f"Episodio {episode} | {args.task}/{args.algo} | {model.device}",
                        flush=True,
                    )
                    while gui.poll() is None:
                        command = keyboard_command(not args.no_keyboard)
                        if command == "q":
                            quit_requested = True
                            break
                        if command == "r":
                            break
                        if command == "p":
                            paused = not paused
                        if paused:
                            with pause_managed_server(args.port):
                                while gui.poll() is None:
                                    command = keyboard_command(not args.no_keyboard)
                                    if command in ("p", "n", "r", "q"):
                                        break
                                    time.sleep(0.05)
                            if command == "q":
                                quit_requested = True
                                break
                            if command == "r":
                                break
                            if command == "p":
                                paused = False
                            if gui.poll() is not None:
                                break
                        # Apply external monitor interventions before observing the policy.
                        env.server.world = env.server.look()
                        env.last_ball = env.world.ball[:2].copy()
                        obs = env.observe()
                        if args.task == "global":
                            if remaining <= 0:
                                option = int(model.predict(obs, deterministic=True)[0])
                                remaining = args.option_steps
                            active = TASKS[option]
                            policy = skills[option]
                            remaining -= 1
                        else:
                            active = args.task
                            policy = model
                        action = int(policy.predict(obs, deterministic=True)[0])
                        _, _, term, trunc, info = env._cycle(action, active)
                        writer.writerow(
                            dict(
                                episodio=episode,
                                paso=env.t,
                                habilidad=active,
                                accion=LABELS[active][action],
                                balon_x=env.world.ball[0],
                                balon_y=env.world.ball[1],
                            )
                        )
                        log.flush()
                        time.sleep(1 / args.fps)
                        if term or trunc:
                            break
                    if quit_requested:
                        break
                    # Keep publishing frames while awaiting the next reset, so the
                    # official monitor remains live and the synchronous watchdog is fed.
                    deadline = time.monotonic() + 2
                    while gui.poll() is None and time.monotonic() < deadline:
                        env.server.tick(["(turn 0)"] * 4)
                        time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
