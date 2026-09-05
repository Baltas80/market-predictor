# Market Predictor

Sistema experimental de investigación cuantitativa para estudiar si los datos de mercado, la macroeconomía y los acontecimientos geopolíticos contienen información útil para predecir movimientos futuros.

> **Aviso:** esto es investigación experimental y no constituye asesoramiento financiero ni garantiza resultados.

## Principios de diseño

1. **Point-in-time:** una observación solo puede usar información que ya estuviera disponible en ese instante.
2. **Sin look-ahead bias:** las etiquetas futuras nunca entran en las variables explicativas.
3. **Backtesting temporal:** entrenamiento, validación y prueba respetan el orden cronológico.
4. **Purga:** cuando el objetivo usa un horizonte futuro, se deja una separación entre entrenamiento y prueba para evitar contaminación de etiquetas.
5. **Lockbox OOS:** la muestra final permanece intacta hasta la evaluación final de A/B/C.
6. **Benchmark obligatorio:** una estrategia predictiva debe superar referencias sencillas antes de considerarse útil.
7. **Reproducibilidad:** código, esquema de datos, procedencia, tests y cambios quedan versionados en Git.

## Arquitectura

```text
Fuentes externas
  ├── Mercado / OHLCV
  ├── Macro / FRED-ALFRED
  ├── GDELT / CAMEO
  └── SEC / registros regulatorios
          │
          ▼
  Ingesta + procedencia + disponibilidad
          │
          ▼
  Dataset point-in-time
          │
          ▼
  Feature engineering
          │
          ├── A: técnico
          ├── B: técnico + macro
          └── C: técnico + macro + eventos
                    │
                    ▼
              Walk-forward + purge
                    │
                    ▼
              Lockbox OOS final
                    │
                    ▼
          Métricas predictivas + backtest
                    │
                    ▼
          Análisis de robustez / significancia
                    │
                    ▼
             D: C + capa de IA
```

## Datos reales

La ingesta reproducible usa varias capas:

- **Mercado:** históricos OHLCV diarios de Stooq; el conjunto inicial recomendado para la validación es el índice S&P 500 (`^spx`). Es un índice de contado, por lo que no debe interpretarse como una serie de ejecución de futuros.
- **Macro:** FRED/ALFRED mediante API, conservando `realtime_start`/`realtime_end` para impedir que revisiones posteriores entren en el pasado.
- **Acontecimientos:** GDELT 2.0 Event Database. Se conservan los campos CAMEO originales y `DATEADDED` como proxy conservador de disponibilidad.
- **Escándalos/fraude/regulación:** SEC Litigation Releases como fuente secundaria específica.

Los datos brutos históricos no se versionan dentro de Git. El repositorio contiene los adaptadores, esquemas, manifiestos y scripts necesarios para reconstruirlos de forma reproducible. FRED requiere una API key y las descargas GDELT son deliberadamente por día para poder reanudar el proceso.

## Ingesta reproducible

Mercado:

```bash
python scripts/ingest_real_data.py --market-symbol '^spx' --start 2000-01-01
```

Macro:

```bash
FRED_API_KEY=TU_CLAVE python scripts/ingest_real_data.py \
  --fred-series UNRATE CPIAUCSL DGS10 DFF T10Y2Y VIXCLS \
  --fred-realtime-start 2000-01-01
```

GDELT:

```bash
python scripts/ingest_real_data.py \
  --gdelt-start 2015-02-19 \
  --gdelt-end 2026-09-04
```

Antes de utilizar los datos para resultados definitivos se debe conservar un manifiesto con fuente, intervalo, fecha/hora de recuperación, regla de disponibilidad y hash de los artefactos generados.

## Motor de acontecimientos

La distinción entre `event_time` y `available_time` es deliberada. Un acontecimiento ocurrido antes de una sesión, pero conocido por el mercado después del cierre, no debe contaminar la predicción de ese cierre.

El motor genera variables de presión temporal y decaimiento exponencial, además de conteos, intensidad, sorpresa y variables específicas por categoría.

La conversión GDELT → esquema interno es conservadora. Se mantienen los códigos CAMEO originales y se asignan categorías amplias solo cuando existe una correspondencia defendible. Las categorías como corrupción, fraude financiero y escándalo corporativo no se inferirán de forma agresiva desde CAMEO: requieren fuentes específicas.

## Backtesting y lockbox

El módulo `backtest.py` implementa ventanas expansivas (*walk-forward*) y un `purge` configurable. Para un horizonte de predicción de `H` observaciones, el pipeline utiliza inicialmente `purge=H`.

La comparación final A/B/C utiliza exactamente el mismo lockbox OOS, purge, umbral y costes financieros. Esto permite medir el valor incremental de macro y acontecimientos sin cambiar las condiciones de evaluación.

No se considerará que existe una ventaja predictiva hasta comprobarla en el lockbox final y después someterla a análisis de robustez.

## Roadmap técnica

### Fase 1 — Fundaciones — COMPLETADA
- [x] Ingeniería de variables OHLCV.
- [x] Objetivo futuro sin etiquetas artificiales al final de la muestra.
- [x] Modelo baseline probabilístico.
- [x] Walk-forward y purga temporal.
- [x] Motor de acontecimientos point-in-time.
- [x] Tests y CI.
- [x] Comparación financiera A/B/C sobre lockbox.

### Fase 2 — Datos reales — EN CURSO
- [x] Adaptador de históricos de mercado.
- [x] Ingesta GDELT por día y reanudable.
- [x] Normalización CAMEO inicial y conservación de códigos originales.
- [x] Deduplicación de acontecimientos por identificador.
- [x] Adaptador FRED/ALFRED con metadatos de tiempo real.
- [x] Registro de procedencia a nivel de fuente y campos originales.
- [x] Script reproducible de descarga/materialización.
- [x] Esquema de fuentes y reglas point-in-time.
- [ ] Ejecutar el histórico completo y generar el dataset de validación OOS.
- [ ] Añadir y validar fuentes específicas para corrupción, fraude y escándalos.
- [ ] Validar horarios exactos de publicación frente a sesiones de mercado.
- [ ] Generar un manifiesto/hash por cada dataset materializado.

### Fase 3 — Investigación estadística — PRÓXIMA
- [ ] Ejecutar A/B/C sobre datos reales.
- [ ] Medir efectos por tipo de crisis.
- [ ] Medir retardos de 1/3/5/10/20/60 sesiones.
- [ ] Intervalos de confianza y bootstrap.
- [ ] Calibración probabilística.
- [ ] Pruebas de robustez por mercado y periodo.
- [ ] Evaluación de estabilidad de las variables y sensibilidad de hiperparámetros.
- [ ] Pruebas contra modelos/benchmarks nulos y permutaciones cuando proceda.

### Fase 4 — Backtest financiero avanzado
- [x] Costes de transacción.
- [x] Slippage.
- [x] Turnover.
- [x] Drawdown máximo.
- [x] Sharpe/Sortino.
- [x] Comparación contra buy-and-hold y benchmarks simples.
- [ ] Coste total desglosado y equity curve reproducible.
- [ ] Sensibilidad a costes y slippage.
- [ ] Análisis de capacidad/turnover.

### Fase 5 — IA experimental — DESPUÉS DE C
- [ ] Definir experimento D = C + IA.
- [ ] Evaluar IA como extractor/estructurador de noticias y acontecimientos, no como oráculo directo de precios.
- [ ] Comparar modelos abiertos/locales bajo el mismo corpus y formato JSON.
- [ ] Candidato inicial: Qwen; comparar con Mistral y otros modelos adecuados.
- [ ] Medir valor incremental exclusivamente en validación temporal y lockbox.
- [ ] Evaluar explicabilidad, coste, latencia y reproducibilidad.
- [ ] Investigar memoria histórica de mercado sin introducir fuga de información.

### Fase 6 — Producción
- [ ] Pipeline automático de datos.
- [ ] Predicciones diarias.
- [ ] Dashboard.
- [ ] Alertas.
- [ ] Monitorización de deriva del modelo.
- [ ] Auditoría de datos y modelos.
- [ ] Registro de predicciones y resultados para evaluación continua.

### Criterio de avance

No se pasa a una fase posterior solo porque el código funcione. Cada salto requiere evidencia reproducible. En particular:

`datos reales → point-in-time → A/B/C → lockbox OOS → robustez estadística → backtest financiero → IA incremental → producción`
