# Market Predictor — Dashboard Design

## Objetivo

Crear una interfaz de investigación que permita distinguir inequívocamente entre datos, resultados OOS y material experimental. El dashboard no debe sugerir capacidad predictiva antes de superar los gates estadísticos y financieros.

## Principios visuales

- Priorizar lectura rápida y trazabilidad sobre decoración.
- Mostrar siempre periodo, universo, versión del dataset, commit y estado del lockbox.
- Separar métricas predictivas de métricas financieras.
- No mostrar una cifra de rentabilidad sin costes y benchmark visibles.
- Estados: `PASS`, `WARN`, `FAIL`, `PENDING`.
- Toda métrica debe poder rastrearse a un artefacto reproducible.

## Estructura

### 1. Executive Research Status

Tarjetas:
- Data coverage
- PIT validation
- A/B/C status
- Lockbox status
- Financial gate
- Last validated commit

Banner superior: `RESEARCH ONLY — NO LIVE TRADING`.

### 2. Historical Coverage

- Línea temporal 2015–2025.
- Cobertura diaria de GDELT.
- Días recuperados.
- Días irreparables.
- Conteo de eventos válidos.
- Indicador de disponibilidad point-in-time.

### 3. A/B/C Comparison

Tabla y gráfico de barras para:
- A: técnico
- B: técnico + macro
- C: técnico + macro + eventos

Métricas:
- ROC-AUC / PR-AUC cuando proceda.
- Log loss.
- Brier score.
- Calibration error.
- Accuracy solo como métrica secundaria.

### 4. Lockbox

Panel bloqueado hasta completar la evaluación:
- OOS period
- observations
- model version
- purge horizon
- threshold
- transaction cost
- slippage
- status

Una vez liberado, mostrar resultados sin permitir modificaciones de parámetros desde la interfaz.

### 5. Financial Results

- Equity curve.
- Drawdown curve.
- CAGR/return cuando sea aplicable.
- Volatility.
- Sharpe.
- Sortino.
- Maximum drawdown.
- Turnover.
- Gross vs net return.
- Transaction costs.
- Slippage.
- Buy-and-hold benchmark.

### 6. Robustness

Controles de lectura:
- Cost sensitivity.
- Slippage sensitivity.
- Threshold sensitivity.
- Walk-forward stability.
- Period stability.
- Null/permutation comparison.
- Bootstrap confidence intervals.

### 7. Provenance / Audit

Mostrar:
- Git commit.
- Dataset manifest hash.
- Source URLs/names.
- Retrieval timestamp.
- Availability rule.
- Feature schema version.
- Experiment configuration hash.
- Lockbox result hash.

## Diseño responsive

Desktop: navegación lateral + contenido de 12 columnas.

Tablet: navegación superior + tarjetas en 2 columnas.

Móvil: una columna, con métricas críticas primero y gráficos apilados.

## Seguridad de interpretación

La interfaz debe bloquear o etiquetar claramente:
- resultados de entrenamiento;
- resultados de validación;
- resultados OOS;
- simulaciones sintéticas;
- resultados financieros no ejecutables;
- experimentos todavía no congelados.

No debe existir un botón que cambie hiperparámetros y simultáneamente presente el resultado como lockbox.

## Fases de implementación

1. Contrato JSON de resultados.
2. Renderer estático reproducible.
3. Equity/drawdown charts.
4. A/B/C comparison.
5. Provenance panel.
6. Lockbox read-only mode.
7. Integración con artefactos CI.
