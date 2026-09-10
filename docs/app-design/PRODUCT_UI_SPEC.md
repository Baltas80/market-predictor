# Market Predictor — App UI Product Specification

## Product direction
A professional mobile dashboard for market research, AI signal analysis, paper trading and controlled execution. The UI must distinguish research, paper trading and live execution at all times.

## Visual system
- Dark-first interface with high information density but generous spacing.
- Neutral graphite background, white/soft-gray typography, restrained semantic green/red/orange status accents.
- Rounded cards, subtle borders, compact charts, no decorative gradients that reduce readability.
- Typography: modern sans-serif; large numeric hierarchy for price, confidence and P/L.
- Primary navigation: Dashboard, Markets, Signals, Agents, Portfolio, Activity, Settings.
- Persistent environment badge: RESEARCH / PAPER / LIVE. LIVE requires explicit confirmation and never appears enabled by default.

## Main screens

### 1. Dashboard
- Header: Market Predictor + environment badge.
- Portfolio equity, daily P/L, exposure, drawdown.
- Market regime card: risk-on / neutral / risk-off with confidence.
- Top signals: asset, direction, confidence, horizon, expected move.
- AI agents summary: ORÁCULO, ATLAS, NEWS AI, GUARDIAN.
- Risk Gate status: operational / blocked / kill switch.
- Recent executions and alerts.

### 2. Market detail
- Asset selector and timeframe.
- Price chart with event markers.
- Trend/momentum/volatility panels.
- Fundamental/macro context.
- Relevant news/events ordered by availability timestamp.
- Model prediction with confidence interval and explicit data timestamp.
- Never display future information as if it were known at prediction time.

### 3. Signals
Each signal card contains:
- Asset and direction.
- Confidence and calibrated probability.
- Prediction horizon.
- Expected return/risk.
- Agent votes and disagreement.
- Evidence/source timestamps.
- Model/version identifier.
- Status: proposed / risk-approved / paper / executed / expired.

### 4. Agents
Agent cards:
- ORÁCULO — macro/regime.
- ATLAS — technical/trend.
- NEWS AI — events/news.
- GUARDIAN — independent risk supervisor.
Each card shows status, latest decision, confidence, timestamp and rationale summary. GUARDIAN cannot be overridden by predictive agents.

### 5. Portfolio
- Equity and cash.
- Positions and average price.
- Exposure by asset/sector.
- Realized/unrealized P/L.
- Risk utilization against immutable limits.
- Drawdown and daily loss.
- Broker connection status.

### 6. Execution
Execution ticket must show:
- Symbol, side, quantity, order type, price/notional.
- Source signal and model version.
- Risk decision and reason.
- Environment.
- Client order ID.
- Broker order ID once submitted.
- State timeline: proposed → risk checked → submitted → accepted → filled/rejected/cancelled.

### 7. Activity / audit
Filterable chronological event log:
- predictions
- agent decisions
- risk decisions
- order submissions
- fills
- cancellations
- broker errors
- kill-switch events
Every event carries timestamp, source/model version and correlation ID.

### 8. Settings / safety
- Environment selector.
- Broker connection.
- Risk limits (display-only unless explicitly unlocked outside AI).
- Kill switch.
- Notifications.
- Data sources.
- Model versions.
- Audit/export.

## Interaction rules
1. AI proposes; Risk Gate authorizes or rejects.
2. No live order is submitted without a visible risk decision.
3. Duplicate client order IDs must not create duplicate broker orders.
4. LIVE mode requires explicit user action and clear visual confirmation.
5. Errors are actionable and never hidden behind generic messages.
6. Every chart/decision displays the relevant timestamp where point-in-time correctness matters.

## Google Play / marketing direction
Primary message: "Research markets. Understand signals. Control execution."
Secondary messages: "Point-in-time data", "AI agents", "Independent risk controls", "Paper trading first".
Avoid claims of guaranteed returns or autonomous profit generation.
