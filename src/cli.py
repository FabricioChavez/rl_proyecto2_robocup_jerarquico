"""Argumentos públicos del proyecto. No lee archivos de configuración."""

import argparse
import os
from pathlib import Path
from rutas import TAREAS, ALGORITMOS, tarea_interna


def comun(parser):
    parser.add_argument(
        "--dispositivo",
        dest="device",
        choices=["cpu", "cuda"],
        default=os.getenv("ROBOCUP_DEVICE", "cpu"),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--horizonte", dest="horizon", type=int, default=200)
    parser.add_argument("--pasos-opcion", dest="option_steps", type=int, default=8)
    parser.add_argument("--puerto", dest="port", type=int, default=6500)
    parser.add_argument(
        "--corrida", dest="run", help="Nombre de la carpeta; por defecto semilla_N"
    )


def entrenamiento(parser, campana=False):
    comun(parser)
    if not campana:
        parser.add_argument(
            "--tarea", dest="task", choices=TAREAS, default="persecucion"
        )
        parser.add_argument(
            "--algoritmo", dest="algo", choices=ALGORITMOS, default="ppo"
        )
        parser.add_argument("--semilla", dest="seed", type=int, default=0)
    else:
        parser.set_defaults(
            task=None, algo=None, seed=None, resume=None, skills=None, skill_algo=None
        )
    parser.add_argument(
        "--escenario", dest="scenario", choices=["1v1", "2v1"], default="2v1"
    )
    parser.add_argument(
        "--pasos",
        dest="steps",
        type=int,
        default=10000,
        help="Decisiones adicionales si se usa --continuar",
    )
    if not campana:
        parser.add_argument(
            "--continuar",
            dest="resume",
            type=Path,
            help="Checkpoint .pth con política y optimizador",
        )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--red",
        dest="network",
        type=int,
        nargs="+",
        help="Capas ocultas; por defecto 128 128 o arquitectura del checkpoint",
    )
    parser.add_argument("--rollout", type=int, help="PPO: 200; A2C: 100")
    parser.add_argument("--batch", type=int, help="PPO: 100; DQN: 128")
    parser.add_argument("--epocas", dest="epochs", type=int, default=5)
    parser.add_argument("--gae-lambda", dest="gae_lambda", type=float, default=0.95)
    parser.add_argument("--entropia", dest="entropy", type=float, default=0.01)
    parser.add_argument("--coef-valor", dest="value_coef", type=float, default=0.5)
    parser.add_argument(
        "--norma-gradiente",
        dest="max_grad_norm",
        type=float,
        help="PPO/A2C: .5; DQN: 10",
    )
    parser.add_argument(
        "--normalizar-ventajas",
        dest="normalize_advantages",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--eps-optimizador", dest="optimizer_eps", type=float)
    parser.add_argument("--rms-alpha", dest="rms_alpha", type=float, default=0.99)
    parser.add_argument("--kl-objetivo", dest="target_kl", type=float)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--buffer", type=int, default=50000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--train-freq", dest="train_freq", type=int, default=4)
    parser.add_argument("--gradient-steps", dest="gradient_steps", type=int, default=1)
    parser.add_argument("--target-update", dest="target_update", type=int, default=500)
    parser.add_argument(
        "--exploration-fraction", dest="exploration_fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--epsilon-final", dest="epsilon_final", type=float, default=0.05
    )
    parser.add_argument(
        "--epsilon-inicial", dest="epsilon_initial", type=float, default=1.0
    )
    parser.add_argument(
        "--epsilon-pasos",
        dest="epsilon_steps",
        type=int,
        help="Por defecto pasos_totales * exploration-fraction",
    )
    parser.add_argument("--shaping", choices=["none", "pbrs", "naive"], default="pbrs")
    parser.add_argument(
        "--premio-exito", dest="success_reward", type=float, default=10.0
    )
    parser.add_argument(
        "--penalizacion-fallo", dest="failure_reward", type=float, default=-5.0
    )
    parser.add_argument("--costo-paso", dest="step_reward", type=float, default=-0.01)
    parser.add_argument("--escala-pbrs", dest="shaping_scale", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-cada",
        dest="checkpoint_every",
        type=int,
        default=1000,
        help="0: solo final.pth; de otro modo se reemplaza checkpoint.pth",
    )
    parser.add_argument("--hilos", dest="threads", type=int, default=2)
    parser.add_argument("--log-cada", dest="log_every", type=int, default=100)
    if not campana:
        parser.add_argument(
            "--habilidades",
            dest="skills",
            type=Path,
            nargs=3,
            metavar=("PERSECUCION", "CONDUCCION", "DEFINICION"),
        )
        parser.add_argument(
            "--algoritmo-habilidades", dest="skill_algo", choices=ALGORITMOS
        )


def parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    root = argparse.ArgumentParser(
        description="RoboCup P2: entrenamiento, evaluación sin ventana, monitor oficial y gráficas.",
        formatter_class=formatter,
    )
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "entrenar", help="Entrenar o continuar desde un .pth", formatter_class=formatter
    )
    entrenamiento(train)
    campaign = commands.add_parser(
        "campana",
        help="Entrenar todas las tareas por algoritmo y semilla",
        formatter_class=formatter,
    )
    entrenamiento(campaign, campana=True)
    campaign.add_argument(
        "--semillas", dest="seeds", type=int, nargs="+", default=[0, 1, 2]
    )
    campaign.add_argument(
        "--algoritmos",
        dest="algorithms",
        choices=ALGORITMOS,
        nargs="+",
        default=list(ALGORITMOS),
    )
    campaign.add_argument(
        "--pasos-jerarquia", dest="hierarchical_steps", type=int, default=1200
    )
    evaluate = commands.add_parser(
        "evaluar",
        help="Evaluación aleatoria por lotes, SIN ventanas",
        formatter_class=formatter,
    )
    comun(evaluate)
    evaluate.add_argument(
        "--tareas", dest="tasks", choices=TAREAS, nargs="+", default=list(TAREAS)
    )
    evaluate.add_argument(
        "--algoritmos",
        dest="algorithms",
        choices=ALGORITMOS,
        nargs="+",
        default=list(ALGORITMOS),
    )
    evaluate.add_argument(
        "--semillas", dest="seeds", type=int, nargs="+", default=[0, 1, 2]
    )
    evaluate.add_argument(
        "--escenarios",
        dest="scenarios",
        choices=["1v1", "2v1"],
        nargs="+",
        default=["1v1", "2v1"],
    )
    evaluate.add_argument("--episodios", dest="episodes", type=int, default=200)
    evaluate.add_argument("--semilla-prueba", dest="test_seed", type=int, default=10000)
    evaluate.add_argument(
        "--velocidad-balon",
        dest="ball_speed",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("MIN", "MAX"),
    )
    evaluate.add_argument(
        "--fuerza-defensor",
        dest="defender_power_range",
        type=int,
        nargs=2,
        default=[100, 100],
        metavar=("MIN", "MAX"),
    )
    evaluate.add_argument(
        "--conducta-defensor",
        dest="defender_behavior",
        choices=["perseguir", "patrulla", "quieto"],
        default="perseguir",
    )
    evaluate.add_argument(
        "--arquero-quieto", dest="static_goalkeeper", action="store_true"
    )
    evaluate.add_argument("--pesos", dest="weights", type=Path)
    evaluate.add_argument(
        "--salida",
        dest="output",
        type=Path,
        help="Cambiar la carpeta de logs de esta evaluación",
    )
    evaluate.add_argument(
        "--sin-graficas",
        dest="no_graphs",
        action="store_true",
        help="Guardar solamente CSV",
    )
    evaluate.add_argument("--ventana", dest="window", type=int, default=20)
    visual = commands.add_parser(
        "ver",
        help="Ejecutar un .pth con rcssmonitor oficial",
        formatter_class=formatter,
    )
    comun(visual)
    visual.add_argument("--tarea", dest="task", choices=TAREAS, default="jerarquico")
    visual.add_argument("--algoritmo", dest="algo", choices=ALGORITMOS, default="ppo")
    visual.add_argument("--semilla", dest="seed", type=int, default=0)
    visual.add_argument(
        "--escenario", dest="scenario", choices=["1v1", "2v1"], default="2v1"
    )
    visual.add_argument("--pesos", dest="weights", type=Path)
    visual.add_argument(
        "--episodios",
        dest="episodes",
        type=int,
        default=0,
        help="0: repetir hasta cerrar",
    )
    visual.add_argument("--fps", type=float, default=10)
    visual.add_argument(
        "--grabar",
        dest="record",
        action="store_true",
        help="Guardar grabación oficial .rcg",
    )
    visual.add_argument("--replay", type=Path)
    visual.add_argument("--salida", dest="output", type=Path)
    visual.add_argument("--sin-teclado", dest="no_keyboard", action="store_true")
    visual.add_argument(
        "--prueba-offscreen",
        dest="offscreen",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    for name in ("balon", "atacante", "companero", "defensor", "arquero"):
        visual.add_argument("--" + name, type=float, nargs=2, metavar=("X", "Y"))
    visual.add_argument(
        "--velocidad-balon",
        dest="ball_velocity",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
    )
    visual.add_argument(
        "--conducta-defensor",
        dest="defender_behavior",
        choices=["perseguir", "patrulla", "quieto"],
        default="perseguir",
    )
    visual.add_argument(
        "--fuerza-defensor", dest="defender_power", type=int, default=100
    )
    visual.add_argument(
        "--arquero-quieto", dest="static_goalkeeper", action="store_true"
    )
    graphs = commands.add_parser(
        "graficar",
        help="Crear gráficas a partir de los CSV existentes",
        formatter_class=formatter,
    )
    graphs.add_argument(
        "--tareas", dest="tasks", choices=TAREAS, nargs="+", default=list(TAREAS)
    )
    graphs.add_argument(
        "--historico",
        dest="historical",
        action="store_true",
        help="Graficar únicamente resultados históricos SB3",
    )
    graphs.add_argument("--ventana", dest="window", type=int, default=20)
    commands.add_parser("estado", help="Ver qué pesos y evaluaciones existen")
    report = commands.add_parser(
        "informe",
        help="Copiar gráficas y tablas a informe/, sin duplicar pesos",
        formatter_class=formatter,
    )
    report.add_argument("--destino", type=Path, default=Path("informe"))
    imitation = commands.add_parser(
        "imitar",
        help="Demostraciones, BC o DAgger con la misma política",
        formatter_class=formatter,
    )
    comun(imitation)
    imitation.add_argument(
        "--tarea", dest="task", choices=TAREAS[:3], default="persecucion"
    )
    imitation.add_argument(
        "--algoritmo", dest="algo", choices=["ppo", "a2c"], default="ppo"
    )
    imitation.add_argument("--semilla", dest="seed", type=int, default=0)
    imitation.add_argument(
        "--escenario", dest="scenario", choices=["1v1", "2v1"], default="2v1"
    )
    imitation.add_argument("--continuar", dest="resume", type=Path)
    imitation.add_argument("--red", dest="network", type=int, nargs="+")
    imitation.add_argument("--lr", type=float, default=3e-4)
    imitation.add_argument("--batch", type=int, default=100)
    imitation.set_defaults(
        rollout=200,
        epochs=1,
        gae_lambda=0.95,
        clip=0.2,
        entropy=0.0,
        value_coef=0.5,
        normalize_advantages=None,
        max_grad_norm=None,
        optimizer_eps=None,
        rms_alpha=0.99,
        target_kl=None,
    )
    imitation.add_argument("mode", choices=["demos", "bc", "dagger"])
    imitation.add_argument(
        "--experto", dest="expert", choices=["humano", "heuristico"], default="humano"
    )
    imitation.add_argument("--datos", dest="dataset", type=Path, required=True)
    imitation.add_argument("--episodios", dest="episodes", type=int, default=5)
    imitation.add_argument("--rondas", dest="rounds", type=int, default=3)
    imitation.add_argument("--epocas-bc", dest="bc_epochs", type=int, default=30)
    imitation.add_argument("--beta", type=float, default=0.2)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if hasattr(args, "gamma") and not 0 < args.gamma <= 1:
        raise ValueError("--gamma debe estar en (0,1]")
    if hasattr(args, "task") and args.task is not None:
        args.task = tarea_interna(args.task)
    if args.command in ("entrenar", "imitar"):
        args.rollout = args.rollout or (200 if args.algo == "ppo" else 100)
        args.batch = args.batch or (100 if args.algo == "ppo" else 128)
    if hasattr(args, "horizon") and (args.horizon <= 0 or args.option_steps <= 0):
        raise ValueError("Horizonte y pasos-opcion deben ser positivos")
    if hasattr(args, "checkpoint_every") and args.checkpoint_every < 0:
        raise ValueError("--checkpoint-cada no puede ser negativo")
    if hasattr(args, "window") and args.window <= 0:
        raise ValueError("--ventana debe ser positiva")
    if getattr(args, "network", None) and any(n <= 0 for n in args.network):
        raise ValueError("--red requiere tamaños positivos")
    if hasattr(args, "lr") and args.lr <= 0:
        raise ValueError("--lr debe ser positivo")
    for name in (
        "log_every",
        "target_update",
        "train_freq",
        "gradient_steps",
        "buffer",
        "batch",
        "rollout",
    ):
        if getattr(args, name, None) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} debe ser positivo")
    if hasattr(args, "gae_lambda") and not 0 <= args.gae_lambda <= 1:
        raise ValueError("--gae-lambda debe estar entre 0 y 1")
    if args.command == "evaluar":
        if not 0 <= args.ball_speed[0] <= args.ball_speed[1] <= 3:
            raise ValueError("--velocidad-balon necesita 0 <= MIN <= MAX <= 3")
        if not 0 <= args.defender_power_range[0] <= args.defender_power_range[1] <= 100:
            raise ValueError("--fuerza-defensor necesita 0 <= MIN <= MAX <= 100")
        if args.run and len(args.seeds) > 1:
            raise ValueError(
                "--corrida identifica un modelo; usa una única --semillas al seleccionarla"
            )
    if args.command == "entrenar":
        from modelo.entrenamiento import ejecutar
    elif args.command == "campana":
        from modelo.entrenamiento import campana as ejecutar
    elif args.command == "evaluar":
        from evaluador.evaluar import ejecutar
    elif args.command == "ver":
        from evaluador.visualizar import ejecutar
    elif args.command == "graficar":
        from evaluador.graficas import ejecutar
    elif args.command == "estado":
        from evaluador.informe import estado as ejecutar
    elif args.command == "informe":
        from evaluador.informe import ejecutar
    else:
        from modelo.imitacion import ejecutar
    return ejecutar(args)
