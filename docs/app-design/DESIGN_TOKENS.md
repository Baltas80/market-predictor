# Market Predictor — Design Tokens

## Color roles
Use semantic roles rather than hard-coded component colors.

- `bg.canvas`: primary dark application background.
- `bg.surface`: elevated card surface.
- `bg.surface-2`: secondary surface.
- `border.default`: subtle structural border.
- `text.primary`: main content.
- `text.secondary`: metadata.
- `status.success`: positive/approved/fill.
- `status.danger`: rejection/loss/error.
- `status.warning`: caution/attention.
- `status.info`: informational state.

Exact color values should be finalized during visual implementation after contrast testing.

## Typography hierarchy
- Display: portfolio equity / headline metric.
- H1: screen title.
- H2: section title.
- Metric: price, P/L, confidence.
- Body: primary explanatory text.
- Caption: timestamps, model versions, source metadata.
- Monospace: order IDs, correlation IDs and technical diagnostics.

## Spacing
Base unit: 4 px. Preferred rhythm: 4 / 8 / 12 / 16 / 24 / 32 px.

## Components
- MetricCard
- SignalCard
- AgentCard
- RiskGateBadge
- EnvironmentBadge
- PriceChart
- EventMarker
- ExecutionTimeline
- PositionRow
- AuditEventRow
- SafetySwitch
- ConfirmationSheet
- EmptyState
- ErrorState

## Accessibility
- Minimum readable contrast target WCAG AA.
- Do not encode financial direction by color alone; pair color with labels/icons.
- Touch targets should be at least 44 dp.
- Support dynamic text sizing without clipping key values.
