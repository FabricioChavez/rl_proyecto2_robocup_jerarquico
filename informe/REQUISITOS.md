# Estado frente al enunciado

- Implementados: tres habilidades, jerarquía, política plana, DQN/PPO/A2C, PBRS y evaluación sin ventana.
- Los 45 modelos y evaluaciones anteriores provienen de SB3 y estan identificados como historicos.
- PPO, A2C y DQN propios en PyTorch pasan 14 pruebas y ensayos cortos CUDA sobre RoboCup real.
- La nueva campana propia parte desde cero; completar sus entrenamientos y evaluaciones no se da por hecho.
- BC/DAgger están preparados; siguen pendientes las demostraciones humanas y su comparación.
- El evaluador admite 1v1/2v1, velocidad de balón y potencia/conducta del defensor por argumentos.
- Las gráficas de aprendizaje, pérdidas, retorno, éxito, goles y duración se generan desde CSV.
- La comparación completa frente al baseline P1, las ablaciones, la superioridad estadística y el informe IEEE final siguen pendientes.
- Completar entrenamiento no demuestra que se alcance el objetivo de 80 % de goles.
- Docker CPU validado: imagen completa, entrenamiento/continuacion, evaluacion, graficas, informe y monitor oficial por noVNC. La imagen CUDA sigue pendiente de construccion y prueba.

La metodología formal está en [METODOLOGIA.md](METODOLOGIA.md).
Los resultados se vuelcan con `python proyecto.py informe`.
