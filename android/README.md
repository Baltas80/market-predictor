# Market Predictor — Android preview

Native Android visual prototype based on the approved Market Predictor UI direction.

## What is included

- Dark-first mobile dashboard.
- RESEARCH/PAPER presentation state (preview is PAPER only).
- Dashboard, Mercados, Agentes IA, Cartera and Más navigation.
- Market cards, signals, agent cards, portfolio positions and Risk Gate status.
- No network calls, broker credentials, account access or real order execution.

## Build locally

From this directory, use Gradle 8.7 with JDK 17:

```text
gradle --no-daemon assembleDebug
```

The debug APK is generated at `app/build/outputs/apk/debug/app-debug.apk`.

## CI artifact

`.github/workflows/android-preview.yml` builds the debug APK on Android-project changes and uploads it as the `market-predictor-preview-apk` Actions artifact. This is a design-preview build, not a production release.
