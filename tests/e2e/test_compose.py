import os

import httpx
import pytest

pytestmark = pytest.mark.e2e
BASE = os.environ.get("URIEL_E2E_URL", "")
ADMIN = {"X-API-Key": "e2e-admin-key"}
FAMILY = {"X-API-Key": "e2e-family-key"}


@pytest.fixture(scope="module")
def http():
    if not BASE:
        pytest.skip("URIEL_E2E_URL not set")
    with httpx.Client(base_url=BASE, timeout=30) as client:
        yield client


def test_admin_gets_gated_tool_answer(http):
    r = http.post("/v1/chat", json={"message": "How is the homelab doing?"}, headers=ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tools_used"] == ["homelab_status"]
    assert "homelab" in body["reply"]


def test_family_member_is_not_offered_admin_tool(http):
    r = http.post("/v1/chat", json={"message": "How is the homelab doing?"}, headers=FAMILY)
    assert r.status_code == 200, r.text
    assert r.json()["tools_used"] == []
    assert "can't access" in r.json()["reply"]


def test_history_persists_and_is_private(http):
    first = http.post("/v1/chat", json={"message": "hello"}, headers=ADMIN).json()
    cid = first["conversation_id"]
    http.post("/v1/chat", json={"message": "hello again", "conversation_id": cid}, headers=ADMIN)
    msgs = http.get(f"/v1/conversations/{cid}/messages", headers=ADMIN).json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert http.get(f"/v1/conversations/{cid}/messages", headers=FAMILY).status_code == 404


def test_unauthenticated_is_rejected(http):
    assert http.post("/v1/chat", json={"message": "hi"}).status_code == 401
