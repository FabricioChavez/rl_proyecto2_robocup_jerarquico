# Revision de futbol: resultados locales y evidencia HFO

Fecha: 19 de noviembre de 2026. Revision local, sin commit ni push.

## Auditoria de los pesos actuales

200 episodios por tarea/algoritmo, semilla de pesos 0; condiciones iniciales 10000-10199, 2v1, horizonte 200, velocidad de balon 0-0.5 y defensor 50-100. Politicas deterministas, sin entrenamiento durante la prueba. Una semilla de pesos; no demuestra superioridad entre algoritmos.

| Tarea | Algoritmo | Exitos / 200 | Goles / 200 | Capturas en menos de 40 ciclos |
|---|---|---:|---:|---:|
| conduccion | a2c | 0 | 5 | - |
| conduccion | dqn | 0 | 1 | - |
| conduccion | ppo | 0 | 9 | - |
| definicion | a2c | 160 | 160 | - |
| definicion | dqn | 161 | 161 | - |
| definicion | ppo | 154 | 154 | - |
| persecucion | a2c | 107 | 1 | 57 |
| persecucion | dqn | 195 | 0 | 192 |
| persecucion | ppo | 149 | 0 | 108 |

Los CSV completos estan en resultados/logs/auditoria_futbol_20261119/.

Persecucion DQN alcanza 192/200 capturas en menos de 40 ciclos. Definicion queda en torno al 80% en su inicio cercano al arco. Conduccion fracasa con los tres metodos; no procede entrenar otro selector sobre esas habilidades.

## Diferencias con la captura del enunciado

- Se uso +10 por exito, -5 por fallo y -0.01 por paso, mas PBRS. No se implementaron literalmente las recompensas de la captura.
- Conduccion usa kick 12 y dash 100, frente a kick 25 y dash 80 de la captura.
- Conduccion empieza con balon x=12.5 y exige x>34: unos 21.5 m de desplazamiento, no mas de 30 m. Tampoco exige mantenimiento continuo del control por todo ese trayecto.
- Definicion no distingue penalizaciones -30 por tiro desviado y -50 por bloqueo del arquero; usa -5 generico.
- El modo 1v1 actual no desactiva al arquero ni al defensor: solo cambia al companero. No sirve como entrenamiento de arco vacio.

## Literatura especifica de futbol

- Hausknecht y Stone: https://arxiv.org/html/1511.04143v5 . PA-DDPG con acciones parametrizadas, aproximacion al balon y avance hacia meta. Experimentos de anotacion en arco vacio; 3 millones de transiciones, aproximadamente 20000 episodios. No equivale a nuestro 2v1 con defensor y arquero.
- Bester, James y Konidaris: https://arxiv.org/html/1905.04388v1 . MP-DQN: media 91.3% en HFO sin arquero, 30000 episodios de entrenamiento; evaluacion sobre 30 agentes y 1000 episodios por agente. Robot Soccer Goal, otro dominio con arquero y aproximacion automatica al balon, obtiene 78.9%. Son dos pruebas diferentes.
- Hausknecht, Chen y Stone: https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/ALA16-hausknecht2.pdf . Imitacion de expertos en HFO. Resultados fuertes en arco vacio, pero imitadores tienen dificultades importantes con arquero. Justifica estudiar BC; no garantiza 80% en nuestro escenario.
- Double DQN: https://arxiv.org/abs/1509.06461 . Reduce sobreestimacion; resultados originales en Atari, no demostracion especifica de nuestro futbol.
- Dueling: https://proceedings.mlr.press/v48/wangf16.html . Arquitectura de valor y ventaja; no demuestra que vaya a reparar nuestra conduccion.

## Recomendacion

Conservar persecucion DQN como candidato y comparar los tiradores en mas semillas. Corregir primero acciones, recompensas, criterio de 30 m y modalidades sin/con oposicion. Despues comparar DQN frente a Double DQN con o sin Dueling, y PPO. Hacer BC/DAgger solo con un experto validado; ajustar con RL. Elegir habilidades por validacion y evaluar su composicion antes de entrenar el selector.

El codigo actual implementa DQN estandar; Double, Dueling y MP-DQN no estan implementados. No se lanzo entrenamiento nuevo ni se modifico codigo de entrenamiento en esta revision.


## Busqueda adicional: drible y Double DQN en RoboCup

- Carvalho y Oliveira, Reinforcement Learning for the Soccer Dribbling Task: https://arthurcarvalho.info/publications/CIG.pdf . SARSA con CMAC en RoboCup, aproximadamente 58% frente a un adversario fuerte. Usa macroacciones de HoldBall y Dribble, construidas con controles existentes; no equivale a aprender cada kick/dash desde cero. La posesion temporal fuera del rango de pateo despues de un toque se distingue de la perdida definitiva frente al rival. Resultado orientativo para diseno del experimento, no benchmark directamente comparable.
- Structure-Preserving Imitation Learning With Delayed Reward (2020): https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00123/full . Usa Double DQN e imitacion de expertos para comportamientos DEFENSIVOS en RoboCup 2D. Encuentra ventajas al mantener la estructura de los modulos expertos y agregar recompensa tardia por goles. Es evidencia de viabilidad de DDQN en RCSS, no prueba de 80% de conduccion.
- R2D-RL (preprint 2026): https://arxiv.org/html/2606.18786v1 . Evalua escenarios de ataque en RoboCup 2D, acciones discretas/parametrizadas, mascaras y shaping. La seleccion del momento de disparo y el diseno del escenario influyen mucho. Una mascara que obliga a tirar incorpora una decision experta; no debe atribuirse al selector como aprendizaje autonomo.

Recomendacion experimental: corregir primero la task; comparar DQN, Double DQN, Dueling DQN y la combinacion Double+Dueling con igual presupuesto, recompensa, dificultad y semillas. Double cambia el objetivo TD; Dueling cambia la arquitectura, por lo que se pueden combinar. No se encontro en esta busqueda una demostracion directamente comparable de que Dueling DQN por si solo resuelva nuestro drible primitivo con defensor.
