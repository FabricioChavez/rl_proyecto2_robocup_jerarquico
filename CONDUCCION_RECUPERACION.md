# Relanzar conducción con recuperación

Cambios locales, sin ejecución de pruebas ni entrenamientos, sin commit ni push.
Los procesos ya arrancados siguen utilizando la versión anterior hasta que los detengas.

```powershell
.\.venv\Scripts\python.exe proyecto.py lote-conduccion --algoritmos ddqn_dueling mpdqn --redes 128x128 --semillas 0 --pasos 100000 --dispositivo cuda --paralelos 2 --lote conduccion_recuperacion --detach
```

Usa un lote nuevo: cambió el contrato y la observación pasa de 36 a 39 componentes,
para incluir estado de recuperación, duración de posesión rival y potencia del defensor.
Los PTH anteriores y su replay no son compatibles. No se borran ni se sobrescriben.

## Cambios

- Un contacto del rival inicia recuperación, no termina inmediatamente el episodio.
- Puede recuperar mediante las acciones existentes: acercarse, girar y patear si alcanza el balón.
  No se añadió `tackle` ni un controlador heurístico que recupere por la red.
- Ventana de recuperación: hasta 40 ciclos sin control y hasta 12 m del balón.
- Si el rival controla el balón durante 12 ciclos consecutivos, termina como fracaso.
- Al recuperar control sin rival más cercano, se sale del modo recuperación y puede seguir conduciendo.
- Se añade un costo de -0,1 por ciclo sin control, además del costo de paso de -0,2.
  No hay un bonus repetible por perder y recuperar el balón.
- Éxito: mismo objetivo de avance neto de 30 m con control al final.
- Currículo: defensor 0 → 40 → 100, pero solo avanza cuando consigue 80% de éxito
  en los últimos 50 episodios, con al menos 50 episodios completados en esa etapa.
  Los umbrales de pasos de la lista `--curriculo` no se usan en el modo `rendimiento`.
- La validación permanece contra el defensor objetivo fijo; sus episodios no hacen avanzar el currículo.
- Logs nuevos: `recuperaciones`, etapa y porcentaje de éxito de la ventana reciente.

Parámetros modificables por CLI: `--gracia-recuperacion`, `--distancia-recuperacion`,
`--posesion-rival-maxima`, `--costo-sin-control`, `--modo-curriculo`,
`--umbral-curriculo`, `--ventana-curriculo`, `--min-episodios-curriculo`.

## Seguir los logs

```powershell
Get-Content resultados/logs/lotes/conduccion_recuperacion/jobs/ddqn_dueling_conduccion_recuperacion_128x128_s0.log -Tail 20 -Wait
```

En otra terminal:

```powershell
Get-Content resultados/logs/lotes/conduccion_recuperacion/jobs/mpdqn_conduccion_recuperacion_128x128_s0.log -Tail 20 -Wait
```

Pesos y CSV siguen en `resultados/modelo_weights/conduccion/ALGORITMO/CORRIDA/`
y `resultados/logs/conduccion/ALGORITMO/CORRIDA/`. `graficar-conduccion` conserva su uso.
Esto habilita aprender recuperación; no demuestra que las nuevas políticas ya la hayan aprendido.

## Ver el modelo en el monitor oficial

Código añadido, sin ejecutar ni validar. Usa una instantánea del checkpoint, no
actualiza automáticamente los pesos mientras el entrenamiento sigue.

```powershell
.\.venv\Scripts\python.exe proyecto.py ver-conduccion --algoritmo mpdqn --corrida conduccion_recuperacion_128x128_s0 --dispositivo cuda
```

Para el otro, cambia a `--algoritmo ddqn_dueling`. Selecciona primero
`checkpoint.pth`, después `final.pth` y finalmente `mejor.pth` si no existe otro.
`--pesos RUTA.pth` permite elegir explícitamente. El defensor usa por defecto la
etapa guardada del entrenamiento; `--defensor 0` lo quita y `--defensor 100`
prueba la dificultad máxima. Los demás criterios permanecen iguales.

El amarillo 1 es el modelo, el azul el defensor. La consola muestra acción exacta,
distancia al balón, avance, retorno y estado de recuperación. Los CSV quedan en
`resultados/logs/visualizacion_conduccion/`, con el checkpoint identificado.
Este comando local abre rcssmonitor mediante WSL en Windows.

Para verlo mediante el navegador, reconstruye la imagen y usa el servicio visual:

```bash
docker build -t robocup-p2:cpu -f docker/Dockerfile .
docker compose run --rm --service-ports visual python proyecto.py ver-conduccion --algoritmo mpdqn --corrida conduccion_recuperacion_128x128_s0
```

Detén antes la sesión visual anterior para liberar el puerto 6080. Abre manualmente
`http://localhost:6080/vnc.html?autoconnect=true&resize=scale`.
