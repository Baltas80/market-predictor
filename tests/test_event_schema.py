import pytest

from market_predictor.event_schema import EventCategory, MarketEvent


def test_event_categories_include_corruption_and_scandals():
    assert EventCategory.CORRUPTION.value == "corruption"
    assert EventCategory.CORPORATE_SCANDAL.value == "corporate_scandal"
    assert EventCategory.POLITICAL_SCANDAL.value == "political_scandal"
    assert EventCategory.FINANCIAL_FRAUD.value == "financial_fraud"


def test_market_event_requires_valid_severity():
    event = MarketEvent("e1", EventCategory.CORRUPTION, "2025-01-01T10:00:00Z", 0.8)
    assert event.severity == 0.8

    with pytest.raises(ValueError, match="severity"):
        MarketEvent("e2", EventCategory.CORRUPTION, "2025-01-01T10:00:00Z", 1.2)
