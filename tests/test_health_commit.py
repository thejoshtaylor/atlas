"""GET /health names the commit the image was built from.

The edge updater (edge/scripts/update.sh) reads the "commit" key and moves
the Pi to that commit. An image built without the GIT_SHA build arg reports
"", and the updater does nothing on an empty value. The route stays open:
these requests carry no cookie, and no admin account exists.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import atlas.app as app_module

_SHA = "0123456789abcdef0123456789abcdef01234567"


def test_health_reports_the_build_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATLAS_GIT_SHA", _SHA)

    response = TestClient(app_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "commit": _SHA}


def test_health_commit_is_empty_without_the_build_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ATLAS_GIT_SHA", raising=False)

    response = TestClient(app_module.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "commit": ""}
