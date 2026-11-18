# Modelos de prueba incluidos

Incluye pesos **semilla 0, escenario 2v1**, para las tres habilidades, la jerarquía
y la política plana, con PPO/A2C/DQN (15 archivos SB3).
Fueron entrenados con CUDA y pueden cargarse para inferencia en CPU.
No requieren reentrenar al clonar el repositorio. Las otras dos semillas
permanecen en los resultados locales de la campaña.

`manifest.json` registra algoritmo, presupuesto y SHA-256 de cada archivo.
Las habilidades/plana recibieron 10 000 decisiones; la jerarquía 1 200 opciones
de hasta 8 ciclos. La finalización del entrenamiento no acredita aprendizaje
satisfactorio ni una tasa de gol de 80 %.

El monitor prefiere pesos locales en `subtareas/` y `global/` si existen;
en una copia nueva usa estos modelos. Los nombres de equipos son neutrales
y se asignan en el entorno, no forman parte de la observación de la red.
