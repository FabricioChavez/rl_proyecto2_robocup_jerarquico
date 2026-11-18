# Mapa del código

La entrada es `../proyecto.py`; los argumentos se declaran en `cli.py`.

| Carpeta/archivo | Responsabilidad |
|---|---|
| `modelo/arquitecturas/ppo.py` | Calcula la pérdida clipped de PPO y optimiza minibatches con Adam |
| `modelo/arquitecturas/a2c.py` | Calcula pérdidas actor/crítico/entropía y actualiza con RMSprop |
| `modelo/arquitecturas/dqn.py` | Calcula objetivo TD y pérdida Huber; actualiza Q y sincroniza la red objetivo |
| `modelo/arquitecturas/redes.py` | Arquitecturas torch.nn actor/crítico y Q |
| `modelo/ventajas.py` | Recurrencia GAE con máscaras de terminal y corte entre episodios |
| `modelo/replay.py` | Buffer circular y muestreo de transiciones DQN |
| `modelo/entrenamiento.py` | Loop compartido, CSV, checkpoint y continuación |
| `modelo/pesos.py` | Formato PTH, optimizador, replay y carga para inferencia |
| `modelo/imitacion.py` | Demostraciones CSV, BC y DAgger |
| `entorno/robocup.py` | Observación, recompensa, PBRS y terminales por tarea |
| `entorno/acciones.py` | Acciones de las tres habilidades y política plana |
| `entorno/cliente_udp.py` | Protocolo oficial UDP |
| `entorno/servidor.py` | Inicia/cierra servidor Linux o WSL |
| `entorno/escenarios.py` | Condiciones compartidas por evaluador y visualizador |
| `evaluador/evaluar.py` | Lotes aleatorios sin monitor; exporta CSV |
| `evaluador/visualizar.py` | Ejecuta los mismos pesos en rcssmonitor |
| `evaluador/graficas.py` | Lee CSV y dibuja PNG |
| `evaluador/informe.py` | Índice de pesos y volcado completo a informe/ |
| `rutas.py` | Única convención de rutas para todos los comandos |

No hay configuraciones de algoritmos en archivos: se reciben desde `cli.py`.
La arquitectura y las formas de los tensores se reconstruyen desde el checkpoint al cargar una política.
No se importan algoritmos externos: [fórmulas e implementación propia](modelo/README.md).
