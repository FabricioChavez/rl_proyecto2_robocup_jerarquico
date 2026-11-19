# Validación Docker CPU — 19 de noviembre de 2026

Construcción completa terminada con:

```bash
docker build -t robocup-p2:cpu -f docker/Dockerfile .
```

Entorno comprobado: Docker Desktop 4.27.1, Engine 25.0.2, Compose 2.24.3,
contenedor Linux amd64, Python 3.11 y PyTorch 2.15.0+cpu.

## Correcciones

- Descargas Debian por HTTPS, cinco reintentos y timeout de 30 segundos.
- Instalación de `libfl-dev`: faltaba `FlexLexer.h` al compilar rcssserver.
- `COPY docker/ /opt/robocup/`: reproduce y evita el fallo de contexto de Compose
  `changes out of order` al copiar un archivo individual dentro de una carpeta.
- El arranque VNC comprueba que escucha el socket. La espera anterior de solo
  0,2 segundos para recibir el banner rechazaba un servidor que sí había arrancado.

## Pruebas completadas

| Prueba dentro del contenedor | Resultado |
|---|---|
| Construcción completa del servidor, monitor y entorno Python | Imagen CPU creada |
| `docker compose run --rm proyecto estado` | Lee los pesos mediante los volúmenes |
| PPO: entrenar 200 decisiones y continuar otras 200 | 400 decisiones, 100 actualizaciones |
| A2C: entrenar 100 decisiones y continuar otras 100 | 200 decisiones, 2 actualizaciones |
| DQN: entrenar 200 decisiones y continuar otras 200 | 400 decisiones, 92 actualizaciones; replay conservado |
| Evaluación de esos tres modelos | CSV por episodio, sin ventanas |
| Generación de gráficas e informe | 7 PNG y volcado de CSV |
| Página noVNC dentro del contenedor | HTTP 200 |
| WebSocket de noVNC hacia VNC | Recibe `RFB 003.008` |
| Monitor oficial con jerarquía PPO, semilla 0 | Episodio completado, salida 0 |

Los ensayos de entrenamiento usan un horizonte corto y prueban ejecución, no calidad
de aprendizaje. Se ejecutaron en un contenedor temporal; los pesos experimentales
de estas pruebas no se mezclan con las corridas de la campaña.

El monitor se ejecutó en Xvfb dentro del contenedor. No se abrió el navegador ni
una ventana en el escritorio del equipo. La imagen CUDA no se construyó en esta validación.

## Probar en la copia local

La imagen CPU ya está construida en la PC donde se realizó esta validación:

```bash
docker compose up visual
```

Abrir manualmente `http://localhost:6080/vnc.html?autoconnect=true&resize=scale`.
En otra PC, construir primero la imagen con el comando del comienzo.
