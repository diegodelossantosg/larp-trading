# News Hunter: hour-by-hour replay, 2016-2026 (informe sencillo)

**Qué se hizo:** se repitió el News Hunter hora a hora durante 10,7 años (mar 2016 - nov 2026) con las noticias en la
hora real en que salieron y el precio de ese momento. Las reglas son las del bot real (caída 2% vs S&P, swing 3%+,
$300M/día o empresa grande, sin volumen alto, sin resultados en 3 días, 10% por compra, 2 al día, máx 30%, stop 20%,
venta a los 10 días), con costes. Sin Claude: compra todo lo que pasa los filtros (la mayor caída primero).
Datos: 124.000 noticias con hora (Alpaca 2016-2026 + todo el mercado desde abril 2026), precios por hora de 924 empresas.
La SEC (8-K) bloquea los servidores de GitHub: no se pudo usar.

## Resultado principal
| | ganancia extra al año sobre el S&P 500 |
|---|---|
| Reglas actuales, todos los datos | +1,2% (6 de 11 años por delante) |
| Reglas actuales, 30 repeticiones con la mitad de las noticias | -0,1% (gana al S&P en 16 de 30) |
| Últimos 6,5 meses con el feed real de ahora (fiscal.ai) | -15% en 6,5 meses (pierde 4% por compra) |

Sin Claude, la regla **empata con el S&P 500** a largo plazo. No hay una ventaja clara.

## 40 variantes probadas (una cosa cambiada cada vez)
Mejores a primera vista, y qué pasó al comprobar si era suerte (30 repeticiones con la mitad de las noticias):
| variante | 1ª vista | comprobación | veredicto |
|---|---|---|---|
| sin stop | +3,0%/año | +1,2%/año mejor, 27 de 30 | única mejora que aguanta (pequeña) |
| stop 30% | +3,4% | +0,6%, 20 de 30 | dudosa |
| sin regla de volumen alto | +0,9% | +0,5%, 22 de 30 | dudosa |
| una acción a la vez (cualquier tamaño) | +4 a +10% | 15 de 30 (moneda al aire) | suerte |
| leer 4 h de noticias | +3,3% | 14 de 30 | suerte |
| 3 compras al día, la más pequeña primero, comprar caídas por resultados | +2% | 11-12 de 30 | suerte |
Peores en todo: vender a 3, 5, 20 o 40 días; stop 10%; caída de 1,5%; volumen alto 1,5x.

## Hallazgo importante: el feed de noticias actual
Desde ~abril 2026 Alpaca usa otro proveedor (fiscal.ai) y solo guarda ~6 meses. Etiqueta mal muchas empresas
extranjeras: Tesco como TSCO (Tractor Supply), Anglo American como AAL (American Airlines), Reckitt como RKT, Vinci
como DG... (40-50% de los titulares de esos símbolos). Con ese feed, la regla sin Claude compra caídas que no tienen
nada que ver con la noticia y pierde. El bot real depende de que Claude detecte estos errores; el gemelo en papel
(paper_hunter.py) no puede.

## Conclusión
1. La regla sola no gana al S&P de forma fiable. Lo que pueda ganar el News Hunter depende del criterio de Claude.
2. La única mejora que resiste las pruebas es quitar el stop del 20% (unos +1%/año), y aun así pequeña.
3. Arreglar la calidad de las noticias (verificar que la empresa del titular es la del símbolo) es más urgente que
   ajustar parámetros.

## Segunda ronda (18 nov): con las reglas nuevas, con Claude y 7 ideas más
**Reglas actuales (sin stop, filtro de nombres):** +3,0%/año sobre el S&P en 10,7 años (8 de 11 años por delante);
con la mitad de las noticias al azar, +1,05%/año y gana al S&P en 25 de 30 repeticiones.

**Con el criterio de Claude** (jul-nov 2026, después de lo que conoce el modelo; sin búsqueda web para no hacer trampa;
119 horas con candidatos): Claude solo eligió 3 compras (la regla sola, 25). Las 2 que pasaron los filtros ganaron
(ASML +4,9%, APP +2,3% sobre el S&P). Total: Claude -1,0% vs S&P -1,7%; regla sola -1,8%. Muy pocas operaciones para
concluir nada; sin búsqueda web Claude casi nunca se atreve ("no puedo confirmar el motivo").
**Filtro de nombres:** en esos mismos meses, la regla sola pasó de -0,65% a -0,10% por compra con el filtro.

**Ideas nuevas (30 mitades, emparejadas con las reglas actuales):** recoger ganancias a +5/+10/+15% (peor / peor / igual),
solo con el S&P subiendo ese día (peor, 2/30), solo con el S&P sobre su media de 200 días (peor, 6/30), solo acciones bajo
su media de 50 días (igual, 11/30), stop 30% en vez de ninguno (peor, 3/30). Ninguna mejora.

**Corrección (18 nov):** las cifras de "mitades" de esta segunda ronda se calcularon antes de tener los nombres de las
empresas (sin el filtro de nombres). Con todo activado: 30 historias con la mitad de las noticias, el bot gana al S&P en
**22 de 30**, de media **+0,85 puntos al año**; cuando pierde, 494 $ menos de media sobre 1.000 $ iniciales en 10,7 años
(peor caso 1.233 $ menos); cuando gana, 831 $ más de media (mejor caso 1.966 $ más). Simulación completa: 5.898 $ frente
a 4.462 $ del S&P (reglas de antes: 5.083 $).

## Tercera ronda (18 nov): ¿se pueden evitar casos como PG&E (2018)?
8 reglas, mismas 30 historias (reglas actuales: 22/30 ganan al S&P, +0,85/año). Arreglan 2018 (palabras de peligro: -2,7 -> -0,6;
no comprar si cayó 10%+ en 5 días: -1,1) pero ninguna mejora el conjunto: mejores que hoy en 4-16 de 30 historias
(moneda al aire o peor), porque también saltan caídas que luego rebotaron. No se aplica ninguna.

## Cuarta ronda (18 nov): ¿con qué frecuencia mirar las noticias?
Corrección: hasta ahora, cuando el bot miraba solo algunas horas ("solo mañanas", "solo tardes"), el simulador tomaba el
precio de otra hora. Arreglado; el resultado de las reglas actuales no cambia (+2,64%/año).
30 historias con la mitad de las noticias (reglas actuales: 22/30 ganan al S&P, +0,85/año):
- cada 2 horas: igual (14/30 mejor); una vez a las 13:00: igual (16/30); una vez a las 11:00: peor (7/30);
  solo mañanas: peor (7/30); solo tardes: peor (10/30).
- cada media hora (precios cada 30 min, comparado con cada hora sobre los mismos datos): no mejora. Cada hora +1,11/año
  (24/30); cada media hora +0,84 (20/30), mejor solo en 13/30; cada media hora leyendo 1 h de noticias: peor (10/30).
Conclusión: mirar cada hora está bien; mirar más a menudo no ayuda. Sin cambios.
