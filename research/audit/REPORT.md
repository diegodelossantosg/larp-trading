# Auditoría grande: qué combinación de agentes sale mejor (2016-2026)

Medida de «mejor»: **cuánto más gana que el S&P 500 con el mismo dinero** (se muestra también la peor caída).
Costes: 0,1 % por compra y por venta. Sin impuestos. Todo reproducible: `PYTHONPATH=. python research/audit/run_ideas.py`, `run_news.py`, `run_combined.py`.

## Qué se probó (≈ 26.000 combinaciones)

| Parte | Qué se cruzó | Datos |
|---|---|---|
| Claude (ideas) | % de la cartera en ideas (0, 10, 30, 50, 80, 100 %), nº de ideas (5, 10, 20), cada cuántos días cambia (semanal, mensual, trimestral), habilidad supuesta (0, +2, +5, +10 %/año) | 2016-2026, grupos A (diseño) / B (validación) / C (final), con empresas desaparecidas |
| Watcher | stops fijos 3, 4, 5, 6, 8, 10, 15, 20 %, sin stop, y 2/3/4/6 veces la oscilación normal diaria, en acciones y en ETFs | igual |
| News Hunter | suelo de liquidez (20 M a 1.000 M$/día), caída mínima (1-4 %), compras/día (1-10), tamaño (2,5-15 %), días de espera (5, 10, 20), con/sin stop | 16.015 noticias 2023-2026, grupos dev / big / holdout |
| Todo junto | stop × % en ideas × habilidad × variante de noticias, en una sola cartera | 2023-07 a 2026-10 |

## Lo que NO se puede probar (y por qué)

- **Claude de verdad.** Conoce lo que pasó y no tengo las noticias de entonces. Probé unas «ideas sin habilidad» (elegidas al azar) y añadí habilidad supuesta. Sirve para saber **cuánta habilidad necesita Claude para compensar**, no para saber si la tiene.
- **2 o 3 decisiones al día.** No hay datos intradía. Se calcula el coste; las pruebas anteriores no encontraron ventaja intradía.
- **Más empresas para el News Hunter.** El fichero de eventos solo tiene ~250 empresas. Bajé el suelo de liquidez, pero para probar de verdad 600 empresas hay que descargar más datos.

## Resultados

### 1. Lo que más dinero da es la regla de SSO (el dinero «parado»)
2017-2026: +5,9 puntos/año sobre el S&P, pero con caída máxima de −47 % (S&P −34 %) y −33 % en 2022 (S&P −18 %). Es el motor de todo, y depende de un mercado alcista de 10 años.

### 2. % en ideas de Claude: más no es mejor, salvo que Claude tenga mucha habilidad
Ideas sin habilidad, 2017-2026, exceso sobre el S&P:

| % en ideas | 0 % | 10 % | 30 % | 50 % | 80 % | 100 % |
|---|---|---|---|---|---|---|
| Exceso (puntos/año) | +5,9 | +4,9 | +2,8 | +0,7 | −2,6 | −4,9 |
| Peor caída | −47 % | −46 % | −44 % | −42 % | −38 % | −36 % |

- Cada 10 % que se mueve de SSO a ideas cuesta ≈ 1,0-1,2 puntos/año si las ideas no son buenas.
- Para que **50 %** compense frente a 0 % hacen falta ≈ **+10 %/año** de habilidad sobre ideas al azar. Con +5 %/año de habilidad, 50 % da +3,6 (todavía menos que 0 %).
- A favor de tener ideas: caídas algo menores y años malos menos duros (2022: −33 % con 0 %, −28 % con 50 %).
- El historial real de Claude (3 semanas, 5 compras) no sirve para juzgarlo.
- **Conclusión: 30 % es el punto medio razonable** (+2,8, 8 de 10 años ganando al S&P) hasta que Claude demuestre habilidad real.

### 3. Frecuencia
- **Cambiar las ideas cada semana pierde 3-4 puntos/año** frente a cada mes o cada trimestre (puro coste). Trimestral ≥ mensual.
- **2 o 3 decisiones al día:** coste extra de 1-5 % al año de la parte de ideas (con 2-5 % de la cartera cambiando en cada decisión), o sea 0,5-2,5 puntos de la cartera con 50 % en ideas, sin ventaja demostrada. **Una vez al día es suficiente.**

### 4. Watcher: los stops no ayudan, los estrechos cuestan dinero
Separado el efecto «tener menos dinero en acciones», comparado con la misma exposición sin stop (puntos/año):

| Stop | Acciones | ETFs |
|---|---|---|
| 3-10 % | −2,1 a −2,6 | −1,1 a −2,0 |
| 6 % (el de ETFs hoy) | −2,6 | −1,1 |
| 4× oscilación (el de acciones hoy) | −1,1 | −1,3 |
| 15 % | −0,9 | −0,4 |
| 20 % | −0,4 | 0,0 |
| Sin stop | 0 | 0 |

Ninguno redujo la peor caída de forma útil. **El mejor es ancho (15-20 %) o ninguno.** Un stop sirve como seguro contra desastres de una sola empresa; con las ideas al azar usadas aquí no se ve ese beneficio, así que un stop ancho es el compromiso.

### 5. News Hunter
- Configuración actual (suelo 300 M$, caída ≥ 2 %, 2 compras/día, 5 %, 10 días): **+0,8 % por compra neta vs S&P (t = 2,3, 410 compras)** al juntar dev + holdout; aporta ≈ **+2 puntos/año** a la cartera tras pagar lo que SSO habría ganado. Por separado, cada grupo sale con t < 1: la señal es real pero débil.
- **Más empresas (suelo 20 M$): peor** (+0,7 en vez de +2,1; negativo en holdout).
- **Más compras al día: no cambia nada** (casi nunca hay más de 2 candidatas).
- **Días de espera: 10 es lo mejor** (5 y 20 peor en 2 de 3 grupos). **Sin stop ≥ con stop.**
- **Tamaño:** más tamaño → más ganancia en los 3 grupos (si la señal es real), pero 10 % y 15 % no salen mejor de forma ordenada en la cartera total (ruido). Razonable: 7,5 %.
- **Elegir las mayores caídas del día** podría explicar la diferencia entre grupos sueltos (+0,3 %) y juntos (+0,8 %): es una hipótesis, no está confirmada.
- Elegir combinaciones en dev y mirar en holdout: correlación de rankings solo 0,24. La mayor parte de las diferencias entre combinaciones es ruido; por eso solo cuento lo que se repite en los grupos.

## El mejor algoritmo según los datos

| Pieza | Hoy | Recomendado | Por qué |
|---|---|---|---|
| Dinero parado | SSO sobre media 200 d, si no SPY | igual | el motor del rendimiento |
| % en ideas de Claude | ≥ 50 % | **30 %** | menos coste mientras no haya habilidad demostrada |
| Decisión de Claude | 1 vez/día | 1 vez/día, cambiar poco | más veces solo suma costes |
| Stop ETFs | 6 % | **15-20 %** | los estrechos cuestan 1-2 puntos/año |
| Stop acciones | 4× oscilación (8-20 %) | 20 % fijo (o sin stop) | el de 4× cuesta ≈ 1 punto/año; 20 % ≈ 0,4 |
| News Hunter | 300 M$, 2 %, 2/día, 5 %, 10 d | **igual, con 7,5 %** | única parte con señal; más empresas empeora |

Estimación 2017-2026 con 30 % en ideas sin habilidad: ≈ +2,8 puntos/año sobre el S&P (+8 de 10 años), más ≈ +2 del News Hunter; caída máxima ≈ −44 %, 2022 ≈ −31 % frente a −18 % del S&P (con 50 % en ideas: −28 %).

## Límites
- Un decenio alcista; la regla de SSO depende de él. En un mercado lateral o bajista largo puede perder mucho más.
- Noticias: solo 3,6 años y ~250 empresas.
- El «stop por oscilación» usa una aproximación (no hay máximos diarios en los datos).
- Muchas combinaciones probadas: el mejor número de una tabla casi siempre está inflado.
- Sin impuestos, sin comisiones de cambio euro/dólar.
