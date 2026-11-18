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
