# Goles en la demo y en la evaluación — 19 de noviembre de 2026

## Resultado confirmado

Se repitieron 200 episodios sin interfaz con la jerarquía PPO propia, semilla de
pesos 0, escenario 2v1, horizonte 200, semillas iniciales 10000 a 10199,
velocidad del balón entre 0 y 0.5 y potencia del defensor entre 50 y 100.
No se modificaron los pesos, la recompensa ni las condiciones para mejorar la tasa.

Se interceptaron los mensajes de árbitro recibidos por los clientes UDP,
además de registrar el indicador `goal` del entorno. Se deduplicaron los mensajes
por su texto y se excluyeron eventos anteriores al ciclo inicial del episodio.

| Medida | Resultado |
|---|---|
| Equipo controlado por la política | `Attack`, lado `l`, confirmado por la respuesta `init` |
| Equipo adversario | `Rival`, lado `r`, confirmado por `init` |
| Portería atacada por `Attack` | Derecha, x = +52.5 |
| Portería atacada por `Rival` | Izquierda, x = -52.5 |
| Episodios con gol de `Attack`, según el árbitro (`goal_l_N`) | 0/200 |
| Episodios con gol de `Rival`, según el árbitro (`goal_r_N`) | 28/200 |
| Episodios con `goal=1` del evaluador | 0/200 |

Los goles observados de `Rival` no se cuentan como éxitos del agente. Los 28
corresponden a eventos recibidos durante los episodios evaluados; no incluyen
una continuación libre de la jugada después del terminal.

Datos: [diagnostico_200_jerarquico_ppo.csv](diagnostico_200_jerarquico_ppo.csv).
Esta es una nueva ejecución de control; el ruido físico del servidor no es
reproducible bit a bit y no debe presentarse como el mismo registro original.

## Qué muestran los registros de la demo

En la sesión local `20261119_055023_241080`, `trayectoria.csv` contiene 283
episodios. Se encontraron 29 episodios con posiciones registradas del balón
fuera de la línea izquierda y dentro del ancho de portería, y ninguno cruzando
la línea derecha. El máximo x registrado fue 48.93 y el mínimo -54.48.

Esos cruces son evidencia geométrica del registro, no un recuento oficial del
árbitro. El CSV antiguo tampoco identifica explícitamente tarea, algoritmo y
pesos, y no registra los ciclos neutros entre episodios. Por eso el control
adicional usa los mensajes oficiales para distinguir a qué equipo pertenece cada gol.

Las condiciones de la demo predeterminada tampoco son idénticas al lote:
usa semillas iniciales desde 0, velocidad de balón 0 y defensor con potencia 100.
El lote indicado usa semillas desde 10000 y aleatoriza velocidad y potencia.

## Estado del aprendizaje

El 0 % a favor no queda contradicho por el marcador visual: el rival también
puede marcar. En este control, el árbitro y el indicador de gol del evaluador
coinciden en cero goles a favor. Esto no constituye una prueba exhaustiva de
todos los casos posibles del detector de goles.

La política jerárquica actual no cumple el requisito de 80 %. En los registros
de entrenamiento PPO de semilla 0, sus últimos 50 episodios también tienen cero
éxitos; conducción tiene cero éxitos en sus últimos 50 episodios y en la pequeña
evaluación existente de 10 episodios. Que definición marque desde su estado
inicial cercano al arco no demuestra que la jerarquía complete el ataque desde
el estado inicial global.

No se aumenta la tasa contando goles del adversario ni sustituyendo el modelo
aprendido por una heurística sin identificarla.
