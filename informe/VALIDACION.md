# Validación de los métodos propios

Se sustituyó la implementación que delegaba PPO/A2C/DQN a Stable Baselines3.
Las clases actuales calculan sus pérdidas y ejecutan backward/step directamente en PyTorch.
Stable Baselines3 se retiró de las dependencias del proyecto.

Comprobaciones realizadas el 18 de noviembre de 2026:

- 14 pruebas automatizadas: protocolo, PBRS, dispositivo, GAE, clipping PPO,
  gradiente A2C, objetivo TD/Huber DQN y continuación de los tres métodos.
- Servidor oficial RoboCup real, RTX 4080 CUDA: PPO 200 decisiones/10 actualizaciones;
  A2C 100/1; DQN 200/42 con warmup reducido. Son ensayos de funcionamiento.
- Los tres PTH propios se cargaron y evaluaron sin monitor en CPU.
- Los 45 pesos históricos SB3 producen probabilidades, valores y Q compatibles
  con las referencias guardadas antes de sustituir la biblioteca (tolerancia rtol 1e-6, atol 1e-7).
- Se inició una campaña propia desde cero en CUDA. Su progreso no se deduce de este
  documento: consultar los CSV y archivos final.pth/checkpoint.pth de cada corrida.
- Las habilidades PPO de semilla 0 tienen una evaluación exploratoria de 10 episodios
  por escenario 1v1/2v1. Una muestra pequeña no demuestra superioridad ni resuelve la jerarquía.

Los entrenamientos históricos y nuevos están separados en pesos, registros y gráficas.
Los datos humanos BC/DAgger siguen pendientes. Docker no se construyó por daemon apagado.

Generar nuevamente el volcado: `python proyecto.py informe`.
