"""Catálogo de demostraciones: un nombre corto -> qué pesos ver en rcssmonitor.

    python proyecto.py demo                 lista el catálogo
    python proyecto.py demo jerarquico      ejecuta una demo (en Docker: ver README)

Todas usan la condición de prueba del informe: 2v1, reglas de recuperación y defensor a
potencia 50. Cada entrada es (descripción, argumentos de 'ver').
"""

from pathlib import Path

W = "resultados/modelo_weights"

CATALOGO = {
    # --- jerarquía completa (Vía A: coordinador sobre 3 habilidades congeladas)
    "jerarquico": ("Mejor sistema: tiro lejano con ventaja + definición entrenada dentro de la "
                   "jerarquía (65,0 % goles)",
                   ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "final4b",
                    "--x-ventaja", "28"]),
    "jerarquico_curriculo": ("Definición con currículo inverso + coordinador DAgger nuevo (51,5 %)",
                             ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "final2"]),
    "jerarquico_informe": ("Sistema del informe: coordinador DAgger + ajuste fino PPO (47,0 %)",
                           ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "final",
                            "--pesos", f"{W}/jerarquico/ppo/final_ft/final.pth"]),
    "jerarquico_dagger": ("Coordinador DAgger del experto programado (44,5 %)",
                          ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "final"]),
    "jerarquico_bc": ("Coordinador solo BC del experto programado (41,0 %)",
                      ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "final",
                       "--pesos", f"{W}/jerarquico/ppo/final/bc.pth"]),
    "jerarquico_humano": ("Primera versión: coordinador DAgger humano (30,5 %)",
                          ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "sistema"]),
    "jerarquico_ppo_rl": ("Jerarquía con todo ajustado por PPO (26,5 %)",
                          ["--tarea", "jerarquico", "--algoritmo", "ppo", "--corrida", "sistema_ft"]),
    "jerarquico_a2c_rl": ("Jerarquía con todo ajustado por A2C (16,0 %)",
                          ["--tarea", "jerarquico", "--algoritmo", "a2c", "--corrida", "sistema_ft"]),
    "jerarquico_dqn_rl": ("Jerarquía con todo ajustado por Double DQN (1,0 %)",
                          ["--tarea", "jerarquico", "--algoritmo", "dqn", "--corrida", "sistema_ft"]),
    "plana": ("Línea base: política plana PPO sin jerarquía",
              ["--tarea", "plana", "--algoritmo", "ppo", "--corrida", "plana50"]),
    # --- persecución
    "persecucion_bc": ("BC humano", ["--tarea", "persecucion", "--algoritmo", "ppo", "--corrida", "humano_bc"]),
    "persecucion_dagger": ("DAgger humano", ["--tarea", "persecucion", "--algoritmo", "ppo", "--corrida", "humano"]),
    "persecucion_bc_rl": ("BC humano + PPO (escalera)", ["--tarea", "persecucion", "--algoritmo", "ppo", "--corrida", "escalera_bc_s0"]),
    "persecucion_ppo": ("PPO desde cero", ["--tarea", "persecucion", "--algoritmo", "ppo", "--corrida", "escalera_s0"]),
    "persecucion_a2c": ("A2C desde cero", ["--tarea", "persecucion", "--algoritmo", "a2c", "--corrida", "escalera_s0"]),
    "persecucion_dqn": ("DQN desde cero", ["--tarea", "persecucion", "--algoritmo", "dqn", "--corrida", "escalera_s0"]),
    "persecucion_ddqn": ("Double DQN desde cero", ["--tarea", "persecucion", "--algoritmo", "dqn", "--corrida", "escalera_ddqn_s0"]),
    "persecucion_ppo_ft": ("Imitación + ajuste PPO", ["--tarea", "persecucion", "--algoritmo", "ppo", "--corrida", "sistema_ft"]),
    "persecucion_a2c_ft": ("Imitación + ajuste A2C", ["--tarea", "persecucion", "--algoritmo", "a2c", "--corrida", "sistema_ft"]),
    "persecucion_dqn_ft": ("Imitación + ajuste DQN", ["--tarea", "persecucion", "--algoritmo", "dqn", "--corrida", "sistema_ft"]),
    # --- conducción
    "conduccion_bc_humano": ("BC humano", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "human_bc"]),
    "conduccion_dagger_humano": ("DAgger humano", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "human_dagger"]),
    "conduccion_bc": ("BC del experto programado", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "expert_bc"]),
    "conduccion_dagger": ("DAgger del experto programado", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "expert_dagger"]),
    "conduccion_bc_rl": ("BC experto + PPO con PBRS", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "hack_pbrs_s0"]),
    "conduccion_bc_ingenuo": ("BC experto + PPO con shaping ingenuo", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "hack_naive_s0"]),
    "conduccion_ppo_ft": ("Imitación + ajuste PPO", ["--tarea", "conduccion", "--algoritmo", "ppo", "--corrida", "sistema_ft"]),
    "conduccion_a2c_ft": ("Imitación + ajuste A2C", ["--tarea", "conduccion", "--algoritmo", "a2c", "--corrida", "sistema_ft"]),
    "conduccion_dqn_ft": ("Imitación + ajuste DQN", ["--tarea", "conduccion", "--algoritmo", "dqn", "--corrida", "sistema_ft"]),
    # --- definición
    "definicion_bc": ("BC humano", ["--tarea", "definicion", "--algoritmo", "ppo", "--corrida", "humano_bc"]),
    "definicion_dagger": ("DAgger humano", ["--tarea", "definicion", "--algoritmo", "ppo", "--corrida", "humano"]),
    "definicion_dagger_rl": ("DAgger humano + PPO desde la zona de tiro (habilidad final)",
                             ["--tarea", "definicion", "--algoritmo", "ppo", "--corrida", "zona_tiro"]),
    "definicion_lejano": ("Definición de 23 acciones entrenada dentro de la jerarquía (la de 'jerarquico')",
                          ["--tarea", "definicion", "--algoritmo", "ppo", "--corrida", "lejano_jer"]),
    "definicion_dqn_zona": ("DAgger humano + Double DQN desde la zona de tiro",
                            ["--tarea", "definicion", "--algoritmo", "dqn", "--corrida", "zona_tiro"]),
    "definicion_ppo_ft": ("Imitación + ajuste PPO", ["--tarea", "definicion", "--algoritmo", "ppo", "--corrida", "sistema_ft"]),
    "definicion_a2c_ft": ("Imitación + ajuste A2C", ["--tarea", "definicion", "--algoritmo", "a2c", "--corrida", "sistema_ft"]),
    "definicion_dqn_ft": ("Imitación + ajuste DQN", ["--tarea", "definicion", "--algoritmo", "dqn", "--corrida", "sistema_ft"]),
}

CONDICION = ["--escenario", "2v1", "--reglas", "recuperacion", "--fuerza-defensor", "50", "--sin-teclado"]


def pesos(argumentos):
    """Ruta del .pth que cargará la demo (para comprobar que existe)."""
    a = dict(zip(argumentos[::2], argumentos[1::2]))
    if "--pesos" in a:
        return Path(a["--pesos"])
    return Path(W) / a["--tarea"] / a["--algoritmo"] / a["--corrida"] / "final.pth"


def listar():
    print("Demos disponibles (python proyecto.py demo NOMBRE):\n")
    for nombre, (texto, argumentos) in CATALOGO.items():
        marca = "" if pesos(argumentos).exists() else "   [faltan pesos]"
        print(f"  {nombre:26s} {texto}{marca}")


def ejecutar(args):
    if not args.nombre:
        return listar()
    if args.nombre not in CATALOGO:
        raise ValueError(f"demo desconocida: {args.nombre}. Lista: python proyecto.py demo")
    from cli import main

    texto, argumentos = CATALOGO[args.nombre]
    print(f"Demo {args.nombre}: {texto}", flush=True)
    return main(["ver", *argumentos, *CONDICION, "--episodios", str(args.episodios)])
