# Market Predictor — Screen Flow

```text
Splash
  ↓
Environment check
  ├── RESEARCH → Dashboard
  ├── PAPER    → Dashboard
  └── LIVE     → explicit safety confirmation → Dashboard

Dashboard
 ├── Market → Market Detail → Signal Detail
 ├── Signals → Signal Detail → Execution Review
 ├── Agents → Agent Detail
 ├── Portfolio → Position Detail
 ├── Activity → Audit Event Detail
 └── Settings → Safety / Broker / Data / Models

Signal Detail
  ↓
Risk assessment
  ├── REJECTED → explanation + no order
  └── APPROVED → Execution Review
                     ↓
              user/environment policy
                     ↓
                  submit
                     ↓
        accepted → filled / cancelled / rejected
                     ↓
                  Activity
```

## Primary user journey
Open app → see regime and portfolio state → inspect a signal → inspect evidence and agent disagreement → inspect independent Risk Gate decision → paper execution → monitor lifecycle → review outcome.

## Safety journey
Any live execution path must visibly expose environment, order notional, buying power, applicable limits, Risk Gate result and confirmation. A kill switch blocks subsequent submissions.
