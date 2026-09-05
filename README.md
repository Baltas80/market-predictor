# Market Predictor

Sistema experimental de investigación cuantitativa para estudiar si los datos de mercado, la macroeconomía y los acontecimientos geopolíticos contienen información útil para predecir movimientos futuros.

> **Aviso:** esto es investigación experimental y no constituye asesoramiento financiero ni garantiza resultados.

## Principios de diseño

1. **Point-in-time:** una observación solo puede utilizar información que ya estuviera disponible en ese instante.
2. **Sin look-ahead bias:** las etiquetas futuras nunca entran en las variables explicativas.
3. **Backtesting temporal:** entrenamiento, validación y prueba respetan el orden cronológico.
4. **Purga:** cuando el objetivo usa un horizonte futuro, se deja una separación entre entrenamiento y prueba para evitar contaminación de etiquetas.
5. **Benchmark obligatorio:** una estrategia predictiva debe superar referencias sencillas antes de considerarse útil.
6. **Reproducibilidad:** código, esquema de datos, tests y cambios quedan versionados en Git.

## Arquitectura

```text
                    ┌── Precios / volumen ──────┐
                    ├── Macro / tipos / inflación│
Fuentes externas ───┼── Eventos geopolíticos ────┼─> Normalización
                    └── Noticias / sentimiento ─┘       │
                                                         ▼
                                                  Feature engineering
                                                         │
                                                         ▼
                                                   Modelo baseline
                                                         │
                                    ┌────────────────────┴──────────────────┐
                                    ▼                                       ▼
                              P(up / down)                         Predicción de retorno
                                    │                                       │
                                    └────────────────────┬──────────────────┘
                                                         ▼
                                                  Walk-forward test
                                                         │
                                                         ▼
                                             Métricas + backtest financiero
```

## Datos reales

La primera ingesta reproducible usa tres capas:

- **Mercado:** históricos OHLCV diarios de Stooq; el conjunto inicial recomendado para la validación es el índice S&P 500 (`^spx`), evitando de entrada el sesgo de supervivencia de una lista de acciones actuales.
- **Macro:** FRED/ALFRED mediante su API, conservando `realtime_start`/`realtime_end` para que las revisiones posteriores no se mezclen con la información disponible en el pasado.
- **Acontecimientos:** GDELT 2.0 Event Database. Se conserva `GlobalEventID`, `SQLDATE`, `EventCode`, `EventBaseCode`, `EventRootCode`, `GoldsteinScale`, métricas de cobertura, `DATEADDED` y `SOURCEURL` antes de convertir al esquema interno.

GDELT publica los eventos en archivos diarios y su Event Database se actualiza cada 15 minutos. `DATEADDED` está en UTC y es el campo apropiado para la resolución temporal de disponibilidad; no se interpreta como la hora exacta de publicación original de la noticia. citeturn1search0turn2view0

FRED/ALFRED permite recuperar observaciones por periodos de tiempo real y por *vintage dates*. Para el backtest se debe usar la versión que estaba disponible en la fecha de decisión, no la revisión actual. citeturn0search1turn0search3turn0search9

### Ingesta reproducible

```bash
python scripts/ingest_real_data.py --market-symbol '^spx' --start 2000-01-01
```

Para macro:

```bash
FRED_API_KEY=TU_CLAVE python scripts/ingest_real_data.py \
  --fred-series UNRATE CPIAUCSL DGS10 DFF T10Y2Y VIXCLS \
  --fred-realtime-start 2000-01-01
```

Para GDELT:

```bash
python scripts/ingest_real_data.py \
  --gdelt-start 2015-02-19 \
  --gdelt-end 2026-09-04
```

La ingesta GDELT es deliberadamente por día y reanudable porque el histórico completo es muy grande. Los datos brutos no se versionan dentro del repositorio: se descargan mediante el script y se guardan en `data/raw/` para evitar convertir Git en un almacén de varios gigabytes.

## Motor de acontecimientos

`events.py` transforma una tabla normalizada de acontecimientos en variables temporales. El esquema mínimo es:

```text
event_time, available_time, event_type, intensity, region,
actor_1, actor_2, source_id, source_url
```

La distinción entre `event_time` y `available_time` es deliberada. Un acontecimiento ocurrido antes de una sesión, pero conocido por el mercado después del cierre, no debe contaminar la predicción de ese cierre.

El motor genera:

- número de acontecimientos recientes;
- número de acontecimientos de conflicto;
- intensidad acumulada;
- intensidad máxima;
- presión de acontecimientos con decaimiento exponencial;
- presión específica de conflicto;
- ventanas de 1, 3, 5, 10 y 20 días.

## CAMEO y categorías

La conversión GDELT → esquema interno es conservadora. Se mantienen los códigos CAMEO originales y se asignan categorías amplias solo cuando existe una correspondencia defendible. Por ejemplo, CAMEO `163` se conserva como **sanctions**; las familias de protesta y conflicto se agrupan en **social_unrest** y **war_conflict**. Los casos ambiguos no se fuerzan artificialmente a corrupción o escándalo: esas categorías requerirán fuentes adicionales específicas, como registros regulatorios y judiciales.

Esto evita presentar una clasificación heurística como si fuera una etiqueta histórica objetiva.

## Backtesting

El módulo `backtest.py` implementa ventanas expansivas (*walk-forward*) y un `purge` configurable. Para un horizonte de predicción de `H` observaciones, el pipeline utiliza inicialmente `purge=H`.

La comparación final A/B/C utiliza el mismo lockbox OOS, el mismo purge y el mismo esquema financiero. Esto permite medir si macro y acontecimientos añaden valor frente al modelo técnico sin cambiar las condiciones de evaluación.

## Estructura actual

```text
market-predictor/
├── .github/workflows/test.yml
├── data/
│   └── events_schema.csv
├── scripts/
│   └── ingest_real_data.py
├── src/market_predictor/
│   ├── backtest.py
│   ├── data_sources.py
│   ├── event_features.py
│   ├── event_io.py
│   ├── event_schema.py
│   ├── experiments.py
│   ├── features.py
│   ├── financial.py
│   ├── model.py
│   ├── pipeline.py
│   └── sources.py
├── tests/
│   ├── test_data_sources.py
│   ├── test_events.py
│   ├── test_features.py
│   └── ...
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Roadmap técnica

### Fase 1 — Fundaciones
- [x] Ingeniería de variables OHLCV.
- [x] Objetivo futuro sin etiquetas artificiales al final de la muestra.
- [x] Modelo baseline probabilístico.
- [x] Walk-forward y purga temporal.
- [x] Motor de acontecimientos point-in-time.
- [x] Tests y CI.

### Fase 2 — Datos reales
- [x] Adaptador de históricos de mercado.
- [x] Ingesta GDELT y normalización CAMEO inicial.
- [x] Deduplicación de acontecimientos por identificador.
- [x] Adaptador FRED/ALFRED con metadatos de tiempo real.
- [x] Registro de procedencia a nivel de fuente y campos originales.
- [ ] Ejecutar el histórico completo y generar el dataset de validación OOS.
- [ ] Añadir fuentes específicas para corrupción, fraude y escándalos.
- [ ] Validar horarios exactos de publicación frente a sesiones de mercado.

### Fase 3 — Investigación
- [ ] Estimar efectos por tipo de crisis.
- [ ] Medir retardos de 1/3/5/10/20/60 sesiones.
- [ ] Comparar modelo técnico vs. técnico+macro vs. técnico+macro+geopolítica.
- [ ] Pruebas de robustez por mercado y periodo.
- [ ] Calibración probabilística.

### Fase 4 — Backtest financiero
- [x] Costes de transacción.
- [x] Slippage.
- [x] Turnover.
- [x] Drawdown máximo.
- [x] Sharpe/Sortino.
- [x] Comparación contra buy-and-hold y benchmarks simples.

### Fase 5 — Producción
- [ ] Pipeline automático de datos.
- [ ] Predicciones diarias.
- [ ] Dashboard.
- [ ] Alertas.
- [ ] Monitorización de deriva del modelo.
