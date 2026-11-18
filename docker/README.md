# Docker

Los comandos completos están en [GUIA.md](../GUIA.md).

- `docker compose build proyecto`: construye la imagen CPU.
- `docker compose run --rm proyecto entrenar ...`: mismo loop de `src/modelo/entrenamiento.py`.
- `docker compose run --rm proyecto evaluar ...`: evaluación sin ventana.
- `docker compose run --rm proyecto graficar`: genera PNG desde CSV.
- `docker compose run --rm proyecto informe`: vuelca CSV/PNG a informe/.
- `docker compose up visual`: monitor oficial por noVNC en localhost:6080.

Para NVIDIA/CUDA antepón `-f docker-compose.yml -f docker-compose.gpu.yml` a Compose.
La base CPU es la alternativa para macOS y máquinas sin NVIDIA.

Solo se montan `resultados/`, `informe/` y `datos/`. Los pesos existentes son PTH.
El servidor y el monitor se ejecutan dentro del mismo contenedor; no hay puertos UDP dinámicos expuestos.
El navegador es únicamente la pantalla remota del monitor oficial.
La construcción completa no se ha ejecutado aquí: el daemon Docker está apagado.
