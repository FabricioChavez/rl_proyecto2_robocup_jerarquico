# Enunciado P2 frente al alcance de esta entrega de código

| Requisito | Implementación / estado |
|---|---|
| Valor frente a Actor-Crítico | DQN, PPO y A2C implementados; campaña corta configurada |
| Tres habilidades | Persecución, conducción/drible/pase y definición, cada una con modelos independientes |
| Meta-controlador | Selección neuronal de habilidades congeladas por 8 ciclos |
| Física oficial | UDP contra rcssserver 19 compilado desde upstream |
| Defensor y arquero activos | Controladores heurísticos ejecutan acciones oficiales |
| 1v1 y 2v1 | Ambos disponibles por CLI; repetir campaña para cada escenario |
| PBRS formal | Potenciales, frontera terminal y prueba en METODOLOGIA.md |
| Reward hacking | Contraejemplo algebraico y campaña none/pbrs/naive disponible; análisis empírico pendiente |
| Aceleración | BC/DAgger implementados; ensayos heurísticos de integración; demostraciones humanas pendientes |
| Baseline P1 | MC 4x5 reimplementado y reentrenable; comparación completa pendiente |
| Tres semillas | Cola usa 0,1,2; esperar resultados antes de afirmar dispersión |
| Curvas y pérdidas | CSV por episodio y actualización; generación de curvas disponible |
| >=200 episodios | Evaluador y campaña configurados; no confundir con prueba corta |
| >=80% gol | NO ACREDITADO hasta evaluar los modelos terminados |
| Superioridad vs plana/heurística | Plana aprendida, heurística, Wilson y bootstrap disponibles; conclusión pendiente |
| Eficiencia muestral y estabilidad | Datos exportados; tabla científica final pendiente |
| Docker | Dockerfile CUDA y compose incluidos; build no validado en esta sesión |
| IEEE <=8 páginas | Pendiente; este Markdown no se presenta como informe final |
| Sustentación/video | Pendiente |

Las instrucciones del enunciado se usan como especificación académica; el alcance
autorizado por la usuaria en esta tarea es implementar las tres habilidades, dejar
BC/DAgger humano preparados, publicar el repositorio e iniciar el entrenamiento
autónomo con CUDA. No se fabrican resultados para completar casillas pendientes.
