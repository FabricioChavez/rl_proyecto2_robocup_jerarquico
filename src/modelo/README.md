# Métodos implementados en PyTorch

No hay una biblioteca de algoritmos RL detrás de estas clases. Gymnasium define únicamente
los espacios y la interfaz del entorno; rcssserver calcula la física. PyTorch proporciona
redes, distribuciones, autograd y optimizadores.

| Archivo | Implementación |
|---|---|
| [arquitecturas/redes.py](arquitecturas/redes.py) | MLP actor/crítico independientes con Tanh; Q y copia objetivo con ReLU |
| [ventajas.py](ventajas.py) | TD residual y recurrencia GAE hacia atrás |
| [arquitecturas/ppo.py](arquitecturas/ppo.py) | PPO con razón de probabilidades, clipping, minibatches y varias épocas |
| [arquitecturas/a2c.py](arquitecturas/a2c.py) | Actor-crítico con ventaja y una actualización por rollout |
| [arquitecturas/dqn.py](arquitecturas/dqn.py) | DQN epsilon-greedy, objetivo Bellman, Huber y sincronización de Q objetivo |
| [replay.py](replay.py) | Buffer circular de transiciones y muestreo aleatorio |
| [entrenamiento.py](entrenamiento.py) | Interacción, resets, rollouts, actualizaciones, CSV y checkpoints |
| [pesos.py](pesos.py) | Guardar/cargar .pth y restaurar optimizador, replay y contadores |

## GAE

`delta_t = r_t + gamma * (1-terminated_t) * V(s_next) - V(s_t)`

`A_t = delta_t + gamma * lambda * (1-boundary_t) * A_next`

Una truncación externa permite bootstrap, pero corta la recurrencia entre episodios.
Un terminal verdadero elimina el bootstrap. En RoboCup el horizonte finito es terminal
y está incluido en la observación. El retorno para entrenar el crítico es `A + V_old`.

## PPO

`ratio = exp(log pi_actual(a|s) - log pi_old(a|s))`

`L_actor = -mean(min(ratio*A, clip(ratio,1-epsilon,1+epsilon)*A))`

`L = L_actor + coef_valor*MSE(V,retorno) - coef_entropia*H(pi)`

Las probabilidades antiguas, ventajas y retornos se recolectan sin gradientes.
Cada época baraja el rollout y actualiza minibatches con Adam. Se limita la norma
del gradiente. `--kl-objetivo` permite detener las épocas antes si la política cambia demasiado.

## A2C

`L_actor = -mean(log pi(a|s) * A)`

Usa la misma pérdida de valor y entropía, sin clipping ni repetición de épocas.
La actualización se hace con RMSprop sobre todo el rollout. Es un actor-crítico
con un único entorno síncrono; no se afirma paralelismo multiworker.

## DQN

`y = r + gamma * (1-terminated) * max_a Q_target(s_next,a)`

`L = mean(Huber(Q_online(s,a), y))`

La acción es aleatoria con probabilidad epsilon; epsilon decrece linealmente.
Las transiciones se guardan en replay. El objetivo se calcula sin gradientes y
la red objetivo está congelada; se copia desde la online cada `--target-update`
transiciones. Es DQN estándar. El optimizador Adam actualiza únicamente Q online.

## Jerarquía y checkpoints

Cada método puede entrenar cualquiera de las tres habilidades, el selector global
o la política plana. El selector usa las habilidades congeladas por `--pasos-opcion`
ciclos. Su descuento es `gamma**pasos_opcion` y su recompensa suma los descuentos primitivos.

Los PTH nuevos llevan `generador='torch_propio'`, `origen_pesos='torch_propio'` y
`formato=2`. Guardan parámetros, estado del optimizador, contadores, RNG y replay DQN.
La continuación reinicia el episodio físico y conserva el aprendizaje; no restaura un rollout parcial.
Los hiperparámetros se vuelven a pasar por CLI. El formato no usa ZIP de modelos ni archivos de configuración.

Los PTH históricos bajo `sb3_*` tienen `generador='sb3_historico'`. Se pueden visualizar
y evaluar con las redes locales, pero no se presentan como entrenamientos propios ni
se usan para inicializar la nueva campaña. Su continuación se rechaza explícitamente.

## Comprobación

Desde la raíz: `python -m pytest -q`. Las pruebas incluyen valores analíticos de GAE,
clipping saturado de PPO, dirección del gradiente A2C, objetivo TD/Huber DQN y
conservación de pesos, optimizador y replay al continuar los tres métodos.
