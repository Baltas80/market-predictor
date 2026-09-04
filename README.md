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

## Datos externos previstos

**GDELT** es una fuente candidata para el componente geopolítico: su Event Database codifica acontecimientos mundiales en más de 300 categorías, con actores, acciones y localización, y GDELT 2.0 se actualiza con alta frecuencia. La fuente se utilizará como materia prima, no como verdad absoluta; habrá normalización, deduplicación y controles de calidad. citeturn0search1turn0search34

**FRED/ALFRED** será una fuente candidata para series macroeconómicas. La API de FRED permite recuperar históricos de series económicas; para backtesting serio debemos conservar la dimensión temporal de publicación/revisión y evitar usar revisiones que no hubieran estado disponibles en la fecha de predicción. citeturn0search0turn0search5

## Backtesting

El módulo `backtest.py` implementa ventanas expansivas (*walk-forward*) y un `purge` configurable. Para un horizonte de predicción de `H` observaciones, el pipeline utiliza inicialmente `purge=H`.

Esto es importante: una división aleatoria de datos financieros puede producir resultados artificialmente buenos porque rompe la estructura temporal.

## Estructura actual

```text
market-predictor/
├── .github/workflows/test.yml
├── data/
│   └── events_schema.csv
├── src/market_predictor/
│   ├── __init__.py
│   ├── backtest.py
│   ├── events.py
│   ├── features.py
│   ├── model.py
│   └── pipeline.py
├── tests/
│   ├── test_events.py
│   └── test_features.py
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
- [ ] Adaptador de históricos de mercado.
- [ ] Ingesta GDELT y normalización CAMEO.
- [ ] Base de acontecimientos con deduplicación.
- [ ] Series macroeconómicas y metadatos de publicación.
- [ ] Registro de procedencia de cada dato.

### Fase 3 — Investigación
- [ ] Estimar efectos por tipo de crisis.
- [ ] Medir retardos de 1/3/5/10/20/60 sesiones.
- [ ] Comparar modelo técnico vs. técnico+macro vs. técnico+macro+geopolítica.
- [ ] Pruebas de robustez por mercado y periodo.
- [ ] Calibración probabilística.

### Fase 4 — Backtest financiero
- [ ] Costes de transacción.
- [ ] Slippage.
- [ ] Turnover.
- [ ] Drawdown máximo.
- [ ] Sharpe/Sortino.
- [ ] Comparación contra buy-and-hold y benchmarks simples.

### Fase 5 — Producción
- [ ] Pipeline automático de datos.
- [ ] Predicciones diarias.
- [ ] Dashboard.
- [ ] Alertas.
- [ ] Monitorización de deriva del modelo.
