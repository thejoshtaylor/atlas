"""Stubs for the `spire_mcp.ha` safety-integration validation map.

Turned green by plan 01-03. Every body raises until then, so the suite
collects cleanly and stays red until its owning task lands.
"""


def test_service_call_allowed_entity():
    raise AssertionError("not implemented: CMD-01")


def test_service_call_denied_entity_never_reaches_ha():
    raise AssertionError("not implemented: SAFE-01")


def test_read_denied_entity_succeeds():
    raise AssertionError("not implemented: SAFE-02")


def test_unresolved_area_target_is_refused():
    raise AssertionError("not implemented: SAFE-01")
