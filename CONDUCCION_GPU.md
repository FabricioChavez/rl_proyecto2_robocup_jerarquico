# Conducción: comandos de los dos experimentos

**Actualización: [recuperación breve y currículo por rendimiento](CONDUCCION_RECUPERACION.md).**
Usa un lote nuevo siguiendo esa guía. Los contratos de pérdida inmediata y currículo
por pasos descritos más abajo corresponden al experimento anterior.

Código local nuevo: **no ejecutado ni validado**, por petición de la usuaria.
No se hicieron commit ni push. No hay garantía de convergencia.
Estos comandos usan el Python local con CUDA de esta PC, no la imagen Docker
anterior. Los nuevos módulos se incorporan a Docker al reconstruir la imagen.

## Lanzar ambos en segundo plano

Desde PowerShell:

```powershell
cd D:\2026_2_CS\RL\PROYECTO\rl_robocup_jerarquico
.\.venv\Scripts\python.exe proyecto.py lote-conduccion --algoritmos ddqn_dueling mpdqn --redes 128x128 --semillas 0 --pasos 100000 --dispositivo cuda --paralelos 2 --lote conduccion_v2 --detach
```

Dos procesos, uno por algoritmo, ambos con redes 128×128 en la misma GPU.
El primero usa Double+Dueling DQN y un menú de ocho acciones; el segundo usa
MP-DQN con tipo de acción discreto y cuatro parámetros continuos: potencia de
dash, ángulo de giro, potencia de kick y ángulo de kick. Cada acción usa solamente
sus parámetros. El critic MP-DQN separa los parámetros por acción mediante pases
enmascarados. El actor usa inversión del gradiente para respetar los límites.

Para comparar también redes 256×128 cambia `--redes 128x128` por
`--redes 128x128 256x128`: serán cuatro entrenamientos, como máximo dos simultáneos.
Para tres semillas añade `--semillas 0 1 2`. `--paralelos 1` los deja en cola.
`--vista-previa` imprime la cola sin iniciar entrenamientos; omite `--detach` al usarla.

No hay fallback a CPU si se solicita CUDA. La física y comunicación del simulador
se ejecutan en CPU; las redes, pérdidas y optimización se ejecutan en CUDA.
100000 son decisiones, no episodios. Solo se entrena conducción, sin selector.

## Ver progreso

| Qué ver | Comando PowerShell |
|---|---|
| Cola y errores de los procesos | `Get-Content resultados/logs/lotes/conduccion_v2/lote.log -Tail 30 -Wait` |
| Reward, actualizaciones y avance de Double+Dueling | `Get-Content resultados/logs/lotes/conduccion_v2/jobs/ddqn_dueling_conduccion_v2_128x128_s0.log -Tail 30 -Wait` |
| Reward, actualizaciones y avance de MP-DQN | `Get-Content resultados/logs/lotes/conduccion_v2/jobs/mpdqn_conduccion_v2_128x128_s0.log -Tail 30 -Wait` |
| Estado de cada ejecución | `Import-Csv resultados/logs/lotes/conduccion_v2/estado.csv` |
| PID del lanzador | `Get-Content resultados/logs/lotes/conduccion_v2/lote.pid` |

Ctrl+C termina el seguimiento de `Get-Content`, no el entrenamiento.
Los logs de cada modelo se crean cuando ese trabajo empieza.

En `resultados/logs/conduccion/ALGORITMO/conduccion_v2_128x128_s0/`:

- `episodios.csv`: recompensa base, recompensa con shaping, éxito, avance y motivo de final.
- `actualizaciones.csv`: pérdidas Q/actor, epsilon y número de actualizaciones.
- `validacion.csv`: rendimiento sin exploración contra defensor con potencia fija 100.
- `validacion_episodios.csv`: detalle por episodio de validación.
- `argumentos.csv`: parámetros exactos del experimento.

La consola imprime progreso cada 500 decisiones. La recompensa completa por episodio
está en `episodios.csv`; no se debe interpretar `loss_q` como recompensa.

## Gráficas durante o después de entrenar

```powershell
.\.venv\Scripts\python.exe proyecto.py graficar-conduccion
```

PNG sin abrir ventanas en `resultados/graficas/conduccion_v2/`.

## Pesos y continuación

En `resultados/modelo_weights/conduccion/ALGORITMO/conduccion_v2_128x128_s0/`:

- `checkpoint.pth`: se reemplaza cada 2000 decisiones, para continuar.
- `mejor.pth`: mejor validación, priorizando éxito y luego avance.
- `final.pth`: pesos al terminar el presupuesto; no equivale a mejor modelo.

Guardan redes, redes objetivo, optimizadores, replay, contadores y RNG.
El episodio físico se reinicia al continuar; no se restaura el simulador bit a bit.
Repite el comando del lote para continuar trabajos interrumpidos hasta el presupuesto
total y omitir los ya terminados. No lo repitas si los entrenamientos siguen activos.
No termina los procesos de modelos al cerrar el lanzador: revisa primero sus logs.

Ejemplo para añadir 20000 decisiones a un entrenamiento individual terminado:

```powershell
.\.venv\Scripts\python.exe proyecto.py entrenar-conduccion --algoritmo mpdqn --red 128 128 --semilla 0 --corrida conduccion_v2_128x128_s0 --pasos 20000 --dispositivo cuda --continuar resultados/modelo_weights/conduccion/mpdqn/conduccion_v2_128x128_s0/final.pth
```

Usa los mismos argumentos del entorno si habías cambiado sus valores.
Los pesos v2 no son compatibles con el selector ni el visualizador antiguo.

## Evaluación independiente: 200 episodios, sin interfaz

```powershell
.\.venv\Scripts\python.exe proyecto.py evaluar-conduccion --pesos resultados/modelo_weights/conduccion/ddqn_dueling/conduccion_v2_128x128_s0/mejor.pth --episodios 200 --semilla-prueba 10000 --dispositivo cuda --puerto 6800
.\.venv\Scripts\python.exe proyecto.py evaluar-conduccion --pesos resultados/modelo_weights/conduccion/mpdqn/conduccion_v2_128x128_s0/mejor.pth --episodios 200 --semilla-prueba 10000 --dispositivo cuda --puerto 6803
```

Las semillas de validación empiezan en 20000; las de prueba, en 10000.
Se evita reemplazar un CSV de prueba existente; cambia `--salida` si deseas repetir.
El criterio es conducción, no goles: al menos 160 éxitos en 200 para alcanzar 80%.

## Contrato compartido del entorno v2

- Misma física oficial, inicio con el balón a 0,5 m del atacante, mismo campo para ambos.
- Éxito: avance neto del balón de 30 m desde el inicio y control recuperado a <=1,085 m.
- Fallo: balón a más de 3 m del atacante, más de 12 ciclos sin control cercano,
  intercepción rival, salida del campo o 300 ciclos agotados. La ventana permite
  recuperar el balón después de un toque; no equivale a posesión instantánea constante.
- Compañero inmóvil y fuera de la jugada; no se cuenta pasarle el balón como conducción.
- Currículo predeterminado: defensor desactivado hasta 20000 decisiones, potencia 40
  hasta 60000 y potencia 100 después. Arquero desactivado. Ambos métodos usan lo mismo.
- Validación siempre contra potencia 100, incluso durante la fase fácil de entrenamiento.
- Recompensa base: +100 por éxito, -50 por intercepción, -30 por otros fallos y -0,2 por ciclo.
- PBRS: potencial = avance/5 - distancia al balón; potencial terminal cero.
  Sus coeficientes son una elección experimental, no valores obligatorios del enunciado.
- Máscara física: no se selecciona kick cuando el balón está fuera del alcance de pateo.
  No se impone una secuencia experta de acciones ni se sustituye a la red al conducir.

Todo se cambia por argumentos: `entrenar-conduccion --help`.
Esto compara **dos enfoques de control con espacios de acción diferentes**; no permite
atribuir cualquier diferencia solo a Double o Dueling.

## Código propio

| Archivo | Responsabilidad |
|---|---|
| `src/entorno/conduccion.py` | Acciones, progreso de 30 m, pérdida de control, recompensas y terminales |
| `src/modelo/arquitecturas/conduccion.py` | Redes Dueling Q, actor de parámetros y critic Multi-Pass |
| `src/modelo/conduccion.py` | Replay, selección de acción, objetivos TD, gradientes y PTH |
| `src/modelo/entrenar_conduccion.py` | Loop, currículo, validación, CSV y continuación |
| `src/experimentos_conduccion.py` | Cola, detach, argumentos, procesos y gráficas |

No usa Stable Baselines3, RLlib ni implementaciones importadas de DQN/MP-DQN.
Usa PyTorch para redes, autograd y optimizadores; NumPy para buffers; Gymnasium solo
para la interfaz del entorno; pandas/Matplotlib para CSV y gráficas.

Referencia de MP-DQN: https://arxiv.org/abs/1905.04388 . Se implementa aquí la idea del
paper; no se importa el código del autor. Los valores por defecto son un experimento
inicial propio, no una reproducción exacta de sus resultados.
