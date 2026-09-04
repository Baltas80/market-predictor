# Market Predictor

Sistema experimental para estudiar y predecir movimientos de mercado combinando:

- histórico de precios y volumen;
- indicadores técnicos;
- eventos macroeconómicos;
- conflictos y crisis geopolíticas;
- sentimiento de noticias;
- modelos estadísticos y de machine learning.

> **Aviso:** el sistema es experimental. Sus predicciones no constituyen asesoramiento financiero ni garantizan resultados.

## Objetivo de la primera versión

Construir un pipeline reproducible que transforme datos históricos en variables explicativas y genere una predicción probabilística del movimiento futuro.

### Arquitectura

```text
Datos de mercado ─┐
Macro ─────────────┼─> Normalización ─> Features ─> Modelo ─> Predicción
Eventos/crisis ────┤                                  │
Noticias/sentimiento┘                                  └─> Backtest
```

## Estructura

```text
market-predictor/
├── src/market_predictor/
│   ├── config.py
│   ├── features.py
│   ├── model.py
│   └── pipeline.py
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

## Próximas fases

1. Ingesta de históricos de acciones e índices.
2. Base de datos de crisis/conflictos con fechas, intensidad, región y tipo.
3. Alineación temporal de mercados y eventos.
4. Features técnicas y de régimen de mercado.
5. Modelo baseline y backtesting sin look-ahead bias.
6. Modelos ML más avanzados.
7. Evaluación fuera de muestra y calibración probabilística.
8. Dashboard y alertas.
