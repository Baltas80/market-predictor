# GDELT 1.0 production lockbox run

The production historical track uses the official GDELT 1.0 daily event stream. GDELT 2.0 remains an independent research track and is not substituted into the production lockbox.

Coverage target for this run: 2015-02-19 through 2025-12-31.

The CI workflow `real-lockbox-gdelt1.yml` performs:

1. Base market/FRED staging.
2. Restartable GDELT 1.0 daily acquisition in 28-day chunks.
3. Gap recovery.
4. Historical staging gate.
5. Point-in-time lockbox A/B/C execution.
6. Full test suite and result artifacts.

This file intentionally contains no generated historical data; raw data remain outside Git and are materialized by CI.
