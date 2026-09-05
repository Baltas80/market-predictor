"""Assembler that guarantees A/B/C share one market observation index."""

from __future__ import annotations

import pandas as pd

from .research_schema import deduplicate_events, validate_market_frame, validate_macro_frame


def assemble_common_observations(
    market: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Validate and return the immutable common observation universe.

    Feature engineering must be applied after this assembly. A/B/C therefore
    begin from exactly the same market timestamps; macro and event information
    can only add columns/information that was available at each timestamp.
    """
    validate_market_frame(market)
    common_market = market.copy()
    common_macro = None
    common_events = None
    if macro is not None:
        validate_macro_frame(macro)
        common_macro = macro.copy()
    if events is not None:
        common_events = deduplicate_events(events)
    return common_market, common_macro, common_events


def align_event_availability(
    events: pd.DataFrame,
    market_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Attach the latest admitted event state to each market decision timestamp.

    This helper intentionally exposes only ``available_at``-eligible rows. It
    does not use event_time or published_at as a substitute for availability.
    """
    if not isinstance(market_index, pd.DatetimeIndex):
        raise TypeError("market_index must be a DatetimeIndex")
    output = []
    event_data = events.copy()
    event_data["available_at"] = pd.to_datetime(event_data["available_at"], utc=True)
    for timestamp in market_index:
        decision = pd.Timestamp(timestamp)
        if decision.tzinfo is None:
            decision = decision.tz_localize("UTC")
        else:
            decision = decision.tz_convert("UTC")
        eligible = event_data.loc[event_data["available_at"] <= decision].copy()
        eligible["decision_time"] = decision
        output.append(eligible)
    if not output:
        return event_data.iloc[0:0].copy()
    return pd.concat(output, ignore_index=True)
