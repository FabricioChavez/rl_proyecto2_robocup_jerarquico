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
La etapa base y la descarga de paquetes Debian por HTTPS se comprobaron dentro de Docker.
La construcción completa sigue pendiente de validación.

## Si falla la construcción

Si Compose muestra `changes out of order` en Windows, construye la misma imagen directamente:

```bash
docker build -t robocup-p2:cpu -f docker/Dockerfile .
```

Después utiliza `docker compose run --rm proyecto ...` o `docker compose up visual`.
El error se ha reportado en Docker Desktop 4.27.1 / Compose 2.24.3; no significa que falte el archivo mencionado.

Si APT muestra `Unable to connect to deb.debian.org` o `connection timed out`,
falló una descarga de dependencias. El Dockerfile usa HTTPS en las dos etapas,
cinco reintentos, timeout de 30 segundos y desactiva el pipelining HTTP.
Conserva la verificación de certificados. Vuelve a ejecutar el comando de construcción.
Estos ajustes permiten reintentar fallos transitorios; no sustituyen una conexión de red disponible.
