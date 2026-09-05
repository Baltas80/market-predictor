"""Compatibility facade for the canonical A/B/C protocol.

The implementation lives exclusively in :mod:`market_predictor.abc_protocol`.
This module remains import-compatible for existing callers while preventing a
second source of truth.
"""

from __future__ import annotations

from .abc_protocol import ABCProtocol, SharedExperimentProtocol

__all__ = [
    "ABCProtocol",
    "SharedExperimentProtocol",
    "assert_common_protocol",
    "assert_oos_configuration_frozen",
]


def assert_common_protocol(protocols: dict[str, SharedExperimentProtocol]) -> None:
    """Compatibility wrapper using the canonical strict A/B/C validator."""
    from .abc_protocol import assert_same_abc_protocol

    assert_same_abc_protocol(protocols)


def assert_oos_configuration_frozen(*, configuration_fingerprint: str, recorded_fingerprint: str) -> None:
    """Require a pre-recorded configuration identity before final OOS use."""
    if not configuration_fingerprint or not recorded_fingerprint:
        raise ValueError("configuration fingerprints are required")
    if configuration_fingerprint != recorded_fingerprint:
        raise ValueError("final OOS configuration differs from the recorded protocol")
