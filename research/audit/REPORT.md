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

## Segunda tabla: sin SSO (dinero parado en SPY, 1×)

SSO es el S&P con el doble de fuerza: gana más cuando el S&P sube sin que haya habilidad. Para medir solo lo que aportan las ideas, las noticias y los stops, se repitió la tabla con el dinero parado en SPY (`run_table_spy.py`, `out_table_spy.csv`, ventana 2023-07 a 2026-10; S&P: +19,3 %/año, caída −18,8 %).

Con ideas de Claude sin habilidad: HOY = +1,7 puntos sobre el S&P; 30 % ideas = +3,3; **0 % ideas + News Hunter = +5,8** (caída −18,6 %, 4 de 4 años ganando); 80 % ideas = −0,7; sin News Hunter = −3,9. Con +5 %/año de habilidad en las ideas: HOY +4,2; 30 % +4,9; 0 % ideas +5,8.
La única fuente de ventaja con respaldo en datos es el **News Hunter** (t ≈ 2,3 con 410 compras; intervalo aproximado del 95 %: de +1 a +9 puntos/año). Las ideas al azar pierden contra el S&P.


## Pruebas adicionales (segunda ronda)

1. **¿Funciona comprar grandes caídas sin mirar las noticias?** (`run_reversal.py`, 10 años, grupos A/B/C, 1.080 combinaciones por grupo). **No**: ventaja ≈ 0 o negativa (−0,1 a −0,3 % por compra). Solo la variante «la acción ya está por debajo de su media de 50 días» sale positiva en los 3 grupos y en las 2 ventanas (+0,3 % por compra, 6 de 6); y con acciones muy líquidas (≥ 1.000 M$/día) y espera de 5 días, las 12 celdas son positivas (+0,13 a +0,85 %). Conclusión: la ventaja del News Hunter viene de la **noticia**, y la «debilidad previa» la refuerza.
2. **Qué noticias rinden más** (`slices_news.py`; dev y holdout por separado, 26 cortes): consistente en ambos grupos: acciones **movidas** (ATR ≥ 4 %: +1,35 y +1,77 %), **debajo de su media de 50 días** (+1,0 y +1,0), **cayendo en los 20 días previos** (+1,0 y +1,4); acciones **tranquilas** (ATR < 2,5 %): negativas (−0,2 y −0,6). Los avisos tipo «analista» o la caída entre 3 y 5 % no se repiten entre grupos.
3. **Cartera con esos filtros** (`run_final2.py`, dinero parado en SPY). El filtro «acciones movidas (ATR ≥ 3 %)» con 10 % por compra sale mejor que la regla anterior en las 4 pruebas:

| Variante | Todas | Solo dev | Solo holdout | Ventaja a la mitad | Peor caída |
|---|---|---|---|---|---|
| Anterior (sin filtro, 7,5 %) | +5,7 | +1,0 | +2,7 | +2,9 | −18,1 % |
| **Acciones movidas, 10 %** | **+7,6** | **+3,1** | **+6,8** | **+3,8** | −20,2 % |
| Movidas y bajo media 50d, 10 % | +4,9 | +2,0 | +5,3 | +2,5 | −19,6 % |
| Bajo media 50d, 10 % | +3,6 | +0,4 | +3,7 | +1,8 | −21,5 % |

Se aplicó «acciones movidas + 10 %». Ojo: se probaron 8 variantes y 26 cortes, así que parte de la mejora puede ser suerte; por eso se exige que se repita en dev y holdout por separado. Combinar «movidas» y «debajo de la media» dobla la ventaja por compra pero reduce las compras a la mitad y no mejora la cartera.


## Tercera ronda: pruebas exhaustivas (T1-T14), con protocolo estricto

Regla de esta ronda: elegir en el grupo de diseño y juzgar en los otros, o la mitad `dev` frente a `holdout`. Todos los scripts: `research/audit/t*.py`.

| Prueba | Qué se probó | Resultado |
|---|---|---|
| T1 | Comprar más tarde que lo ideal; costes de 0,1 % a 1 % | La ventaja se concentra en la **primera hora** tras la noticia (+0,9/+0,5 % a la hora; +0,3/+0,2 % al cierre). Aguanta hasta ~0,3 % de coste por operación; con 1 % desaparece |
| T2 | Prueba de suerte (20.000 sorteos de días al azar) | Regla actual p = 0,03; movidas p = 0,04; movidas + bajo su media de 50 d p = 0,008. **Ninguna llega a p < 0,0015** que pediría haber probado ~34 reglas: sugerente, **no probada** |
| T3-T5 | Compra de «capitulación» sin noticias (caída + volumen disparado + muy líquida), 10 años | Con todo el análisis hecho solo con datos del día anterior y 3 grupos, aparentemente +1 a +6 puntos/año |
| T7-T8 | Misma regla, sin fuga de información y con el protocolo estricto (elegir en A, juzgar en B y C) | **No generaliza**: la mejor en A (+8,4) da −1,7 en B y −0,6 en C; solo el 20 % de las combinaciones gana en los 3 grupos. **Descartada** |
| T6 | Plazos, stops, tamaños y universo de esa regla | Sin ventaja robusta; los stops vuelven a restar |
| T9 | Timing del S&P sin apalancamiento (medias de 50/100/200 días, volatilidad, calendario, noche/día) | **Ninguna** gana al S&P comprado y mantenido; solo reducen la caída a costa de rendimiento |
| T9 | 10 factores de selección de acciones, elegidos en A y juzgados en B y C | Ninguno aguanta (los 5 mejores de A: +1,3 en B, −4,1 en C) |
| T10 | Tipos de titular, hora del día, noticias positivas, caídas tras resultados (dev → holdout) | **Comprar subidas no da ventaja** (−0,02/−0,12). Lo demás: escaso o inconsistente |
| T11 | Inclinaciones a otros ETFs frente a SPY | SMH, XLK, QQQ ganaron, pero con caídas de −45/−34/−35 % y peores años de −15 puntos: apuestas de estilo, no habilidad |
| T12 | Suelo de liquidez con acciones movidas; años necesarios para confirmar la ventaja | Más liquidez → más ventaja por compra en ambos grupos (dev +0,9 → +2,1 → +3,2; holdout +0,5 → +1,5 → +1,4 con 300 → 600 → 1.000 M$). **Con dinero de práctica harían falta ~5 años (≈ 60 compras/año) o ~20 si la ventaja real es la mitad**: la confirmación tiene que venir de datos históricos |
| T13 | Cartera con suelos 300/600/1.000 M$ y tamaños 10/15/20 % | 600 M$ es el mejor: 10 % → +9,3 (todas), +4,4 dev, +5,0 holdout, +4,6 mitad; 15 % → +12,2 / +10,1 / +8,2 / +6,0. 1.000 M$ deja pocas compras y baja a +2 a +4. Es una zona con forma de «U invertida», pero se vio con los dos grupos a la vez: queda pendiente la comprobación con años nuevos |
| T14 | Dinero real en España (aritmética) | Con ventaja de 6 puntos antes de impuestos queda +0,7 a +3 puntos después de impuestos y estructura; con 3 puntos queda en cero o negativa |

Conclusión provisional: **el News Hunter sobre acciones movidas (y probablemente con suelo de 600 M$) es lo único con una señal medida, pero débil y no confirmada**. Todo lo demás probado (timing, factores, capitulación, ETFs, noticias positivas, salidas anticipadas) no mejora al S&P.

## Cuarta ronda: las descargas grandes (T15)

Dos ficheros nuevos de noticias, con el dinero parado en SPY (así no hay apalancamiento que ayude):
`events_long` (2018-2026, ~190 empresas) y `events_wide` (2023-2026, ~530 empresas). Script: `t15_long_wide.py`, salida `out_t15.csv`.

| Regla actual (volátiles ATR 3%+, 10% por compra, 2 al día) | Ganancia extra sobre el S&P por año | ¿Es suerte? (t) |
|---|---|---|
| Larga, antes de jul-2022 (días nuevos, incluye crash 2020 y 2022) | +2,4 a +3,5 puntos | t 0,4 a 0,8 (no se distingue de la suerte) |
| Larga, después de jul-2022 | +1,4 a +2,9 puntos | t 0,2 a 0,5 |
| Ancha, 2023-2026 (más empresas) | +6,7 a +7,7 puntos | t 1,3 a 1,8 |

- La regla actual sale positiva en los tres cortes, pero ninguno es estadísticamente sólido. Mejor estimación honesta: +2 a +4 puntos al año, con mucho margen de error.
- El suelo de liquidez de 600 M NO aguanta: en los días nuevos (antes de jul-2022) sale -0,4 a -0,9 puntos. Solo funcionaba en los años que ya habíamos mirado. Descartado, no se aplica.
- Conclusión: se dejan los ajustes actuales. No hay evidencia nueva para cambiar tamaño ni suelo.
