# MDP, algoritmos y jerarquía

## Estado y acciones

La física proviene de `rcssserver`, no del entorno cinemático del notebook P1.
La observación de 34 componentes contiene: distancia y seno/coseno del ángulo
relativo hacia balón, ambos postes, defensor, compañero y arquero (18);
posición, velocidad y seno/coseno de orientación del atacante (6);
velocidad del balón (2); estamina y tiempo restante implícito (2);
velocidades de defensor, arquero y compañero (6). Las coordenadas de base se
leen del trainer, por lo que es un MDP con estado privilegiado, no una solución
de percepción parcial. La observación se normaliza por las dimensiones del campo.

Persecución tiene 4 acciones: dash 100, dash 50, turn +35 y turn -35 grados.
Conducción tiene 7: dash, turn +/-35, drible centro/arriba/abajo y pase al compañero.
Definición tiene 5: kick 100 hacia y = -6, -3, 0, 3, 6 en x = 52.5.
Las direcciones de kick se calculan en coordenadas relativas al cuerpo.
La política plana reúne las 13 primitivas únicas; la jerarquía selecciona entre
las tres políticas aprendidas y congeladas durante 8 ciclos.

En 1v1 el compañero no participa y permanece fuera del ataque. En 2v1 ejecuta
apoyo y tiro heurísticos. Defensor y arquero están activos en todos los casos.
El arquero puede ejecutar `catch` dentro del área; el defensor persigue y despeja.
La tasa de éxito de una habilidad no se usa como tasa de gol del ataque global.

## Recompensa base y horizonte

R = +10 al cumplir la subtarea; -5 al fallar o agotar el horizonte; -0.01 en otro
caso. Persecución exige distancia <0.8 m. Conducción exige balón más allá de x=34
y posesión del atacante o del compañero participante. Definición y global exigen
cruzar x=52.5 dentro de los postes; se interpola y en el segmento de cruce.
La intercepción por rival, salida del balón o fin de horizonte termina el episodio.
El horizonte predeterminado de entrenamiento/evaluación es de 200 ciclos físicos.
El tiempo se incluye en el estado y este límite es terminal del problema finito,
no una truncación externa con bootstrapping. Gamma primitivo = 0.99.

## PBRS

Potenciales acotados en el dominio del episodio:

- persecución: Phi(s) = -distancia(atacante, balón)/10;
- conducción: Phi(s) = (x_balón - 52.5)/20 - distancia(atacante, balón)/20;
- definición/global/plana: Phi(s) = -distancia(balón, centro del arco)/20.

En cualquier estado terminal, incluido el horizonte finito, Phi = 0. Se aplica:

`F(s,a,s') = gamma * Phi(s') - Phi(s)`.

Para una trayectoria de longitud T:

`sum_t gamma^t F_t = -Phi(s_0) + gamma^T Phi(s_T) = -Phi(s_0)`.

Por tanto `G_shaped = G_base - Phi(s_0)`. Para cada estado/tiempo inicial fijo,
el término adicional no depende de las decisiones posteriores:
`Q*_shaped(s,a) = Q*_base(s,a) - Phi(s)`. El argmax y la política óptima se
conservan bajo los supuestos anteriores. Premiar solo el desplazamiento positivo
del balón no tiene esa propiedad y puede premiar oscilaciones sin progreso neto.
`scripts/pbrs_ablation.py` reproduce ese contraejemplo; `--ablations` prepara
experimentos de aprendizaje reales con none/PBRS/naive.

Las recompensas de una opción se acumulan con descuento primitivo. Puesto que
la duración es fija en 8 ciclos salvo al terminar el episodio, el meta-controlador
usa `gamma_meta = gamma^8`. Así se conserva la identidad telescópica de PBRS.

## Algoritmos

Se usa Stable-Baselines3 2.7.1, con red MLP 128x128 en CUDA. DQN implementa
replay, red objetivo y target `r + gamma * max_a Q_target(s',a)`; no se presenta
como Double DQN. Buffer 50 000, batch 128, warm-up 500, actualización cada cuatro
decisiones, sincronización objetivo cada 500. PPO: clipping 0.2, GAE lambda .95,
rollouts 200, batch 100 y cinco épocas. A2C: rollouts 100 y lambda .95.
PPO/A2C usan entropía .01. Learning rate compartido 3e-4.

La versión tabular es una nueva implementación MC first-visit con 20 estados y
cuatro acciones inspirada en P1, reentrenada en el entorno actual; no acredita
igualdad de dificultad ni de resultados con el simulador antiguo.

## Imitación pendiente de demostraciones humanas

BC: entropía cruzada `-log pi(a_experta|s)` sobre la misma política PPO/A2C.
DAgger: etiquetado de estados visitados por la aprendiz, mezcla de acciones con
probabilidad beta de experta, agregación de datos y nuevo ajuste por ronda.
Se conserva el origen `human` o `heuristic` por muestra. Tras BC/DAgger se retoma
el entrenamiento RL con `--resume`, permitiendo aprender también el crítico.
La interfaz humana congela el proceso del servidor mientras espera la etiqueta.

## Validez experimental

Separar entrenamiento de evaluación: semillas iniciales 0/1/2 para entrenar,
estados iniciales 10000+episodio para probar. Cada evaluación reinicia el episodio
y no actualiza pesos. Se guardan al menos 200 episodios por modelo en la campaña.
La tasa de gol y los retornos base se calculan sin sumar shaping. Los intervalos
Wilson describen incertidumbre binomial dentro de una evaluación. El bootstrap
de diferencias no sustituye replicación entre semillas ni corrige múltiples
comparaciones. No se afirma éxito si los resultados no lo demuestran.

El RNG físico de upstream rcssserver 19 se inicializa con el reloj y la opción
CLI `random_seed` no está habilitada. La repetibilidad está controlada en el
aprendiz y en las condiciones iniciales, pero no es bit a bit. Esto también
impide describir evaluaciones como perfectamente emparejadas en ruido físico.

Limitaciones: la conducción usa acciones discretas sobre física continua; no hay
entrenamiento multiagente del compañero; las redes de definición emiten tiros
y esperan su resultado sin perseguir rebotes; no hay transferencia automática
de una política de visión privilegiada a sensores parciales. La campaña corta
sirve de punto de partida y no garantiza resolver la tarea con 10 000 pasos.
