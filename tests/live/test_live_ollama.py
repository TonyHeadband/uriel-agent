import os

import httpx
import pytest

pytestmark = pytest.mark.live
BASE = os.environ.get("URIEL_LIVE_URL", "")


@pytest.fixture(scope="module")
def http():
    if not BASE:
        pytest.skip("URIEL_LIVE_URL not set")
    with httpx.Client(base_url=BASE, timeout=180) as client:
        yield client


def test_real_model_uses_gated_tool_for_admin(http):
    r = http.post(
        "/v1/chat",
        json={"message": "What's the current status of my homelab servers?"},
        headers={"X-API-Key": os.environ.get("URIEL_LIVE_ADMIN_KEY", "e2e-admin-key")},
    )
    assert r.status_code == 200, r.text
    assert "homelab_status" in r.json()["tools_used"]


def test_real_model_answers_small_talk_without_tools(http):
    r = http.post(
        "/v1/chat",
        json={"message": "Good morning! How are you?"},
        headers={"X-API-Key": os.environ.get("URIEL_LIVE_ADMIN_KEY", "e2e-admin-key")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["tools_used"] == []
