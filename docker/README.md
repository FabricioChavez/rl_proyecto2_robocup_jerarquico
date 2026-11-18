# Docker: simulación, políticas y monitor oficial

## Arquitectura

El contenedor contiene `rcssserver`, PyTorch/SB3 y el **rcssmonitor oficial**.
Xvfb proporciona una pantalla virtual; x11vnc/noVNC muestran esa pantalla en el
navegador. No se usa la instalación Qt/WSLg del host y no se ha creado un panel
de fútbol alternativo. El servidor UDP y el monitor se comunican por localhost
dentro del mismo contenedor: no hay que publicar puertos UDP dinámicos.

La base es una imagen Linux multi-arquitectura de Python, con PyTorch CPU.
Está preparada para Docker en Windows, Linux y macOS; el build ARM64 no fue
ejecutado aquí. La variante CUDA requiere NVIDIA y un runtime Docker compatible.
No se ofrece aceleración de GPU Apple mediante Metal dentro del contenedor.

## Arrancar la parte visual

Inicia Docker tú y abre una terminal en la raíz del proyecto:

```bash
docker compose up --build visual
```

Abre [noVNC local](http://localhost:6080/vnc.html?autoconnect=true&resize=scale).
Verás la ventana del monitor oficial, con la jerarquía PPO. El puerto web está
publicado únicamente en `127.0.0.1`. No expongas esta configuración sin contraseña
a una red externa; está destinada a uso local.

Con GPU NVIDIA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build visual
```

Windows necesita el backend WSL2 de Docker Desktop. Linux necesita NVIDIA
Container Toolkit. La imagen CUDA usa ruedas PyTorch cu128; una GPU visible
sin driver compatible no basta. Si CUDA se solicita pero no está disponible,
el programa falla con un error explícito.

No se abren navegadores ni ventanas del host desde los scripts.

## Cambiar la política o el escenario

Detén la sesión anterior antes de reutilizar el puerto 6080:

```bash
docker compose down
docker compose run --rm --service-ports visual python -u -m robocup_rl.monitor --task persecucion --algo dqn --device cpu --no-keyboard
```

Opciones de `--task`: `persecucion`, `conduccion`, `definicion`, `global`, `plana`.
Opciones de `--algo`: `ppo`, `a2c`, `dqn`. Cambia el modelo con `--seed`;
los modelos incluidos son semilla 0 y escenario 2v1.

Para CUDA, añade los dos `-f` al comando Compose y utiliza `--device cuda`.
La configuración GPU utiliza una etiqueta de imagen distinta de la CPU.

Condiciones cambiantes:

```bash
docker compose run --rm --service-ports visual python -u -m robocup_rl.monitor --task persecucion --algo ppo --device cpu --scene escenarios/monitor_cambiante.json --no-keyboard
```

Edita ese JSON para fijar posiciones, velocidades y conducta del defensor.
Los controles visuales nativos del monitor siguen disponibles: clic derecho
`Drop Ball`, tiros libres, menú `Referee` y opciones de `View`.
Los equipos se llaman `Attack` y `Rival`.

Para congelar/reanudar una sesión iniciada con `up`:

```bash
docker compose pause visual
docker compose unpause visual
```

Esto pausa el contenedor completo, incluida la conexión gráfica; la última
imagen permanece y la conexión se recupera al reanudar. Las teclas p/n/r de la
ejecución nativa no se envían al agente desde el escritorio remoto.

## Entrenar dentro de Docker

CPU (también disponible en macOS):

```bash
docker compose --profile training run --rm entrenamiento
```

CUDA con NVIDIA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile training run --rm entrenamiento
```

Una sola subtarea:

```bash
docker compose --profile training run --rm entrenamiento python -m robocup_rl.ejecutar --task conduccion --algo ppo --steps 10000 --device cpu
```

Los resultados se guardan en las carpetas montadas del host, y no se pierden
al destruir el contenedor. La campaña omite modelos/evaluaciones ya terminados.

## Modelos y grabaciones

La imagen incluye los pesos de `pretrained/`. Si hay pesos locales terminados
en `subtareas/` o `global/`, el monitor los prefiere. El formato SB3 carga los
pesos en el dispositivo seleccionado; no hace falta reentrenar para CPU.

Las grabaciones oficiales `.rcg` se escriben en `runs/monitor/` del host.
Para abrir una grabación en el monitor del contenedor:

```bash
docker compose run --rm --service-ports visual python -m robocup_rl.monitor --replay runs/monitor/RUTA_DE_SESION/partida.rcg
```

## Qué se validó y qué falta

- Se validaron y fusionaron las configuraciones CPU y CUDA con Compose v2.24.3.
- Se cargaron los modelos entrenados y se ejecutó la jerarquía real en CPU.
- Se comprobó una grabación nueva con nombres neutrales.
- Se arrancó la pantalla virtual y se validó HTTP de noVNC y su puente WebSocket
  hasta el protocolo RFB, sin abrir navegador ni escritorio del host.
- No se ejecutó `docker build`: el daemon de esta PC está apagado. Tampoco se
  acreditan ejecuciones completas en macOS/ARM64 o con CUDA dentro del contenedor.

Requisitos y referencias: [GPU en Compose](https://docs.docker.com/compose/how-tos/gpu-support/),
[GPU en Docker Desktop Windows](https://docs.docker.com/desktop/features/gpu/),
[noVNC oficial](https://github.com/novnc/noVNC).
