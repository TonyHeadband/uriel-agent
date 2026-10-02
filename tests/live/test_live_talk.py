import asyncio
import os
import time

import pytest

from uriel.gateway.talk_api import TALK_USER, TalkClient

pytestmark = pytest.mark.live
NC = os.environ.get("URIEL_LIVE_NC_URL", "")


async def test_uriel_can_list_its_talk_rooms():
    password = os.environ.get("URIEL_NC_APP_PASSWORD")
    if not (NC and password):
        pytest.skip("URIEL_LIVE_NC_URL and URIEL_NC_APP_PASSWORD not set")
    client = TalkClient(NC, TALK_USER, password)
    try:
        rooms = await client.rooms()
    finally:
        await client.aclose()
    assert all(r.token for r in rooms)


async def test_a_dm_round_trip_as_a_test_user():
    # Needs a gateway with URIEL_TALK_ENABLED polling this Nextcloud, and a test user in lldap's family group.
    user, password = os.environ.get("URIEL_LIVE_TALK_USER"), os.environ.get("URIEL_LIVE_TALK_PASSWORD")
    if not (NC and user and password):
        pytest.skip("URIEL_LIVE_NC_URL, URIEL_LIVE_TALK_USER and URIEL_LIVE_TALK_PASSWORD not set")
    person = TalkClient(NC, user, password)
    try:
        token = await person.open_dm(TALK_USER)
        sent = await person.post(token, "Live test: please just say hi back.")
        replies, deadline = [], time.monotonic() + 180
        while not replies and time.monotonic() < deadline:
            await asyncio.sleep(3)
            replies = [m for m in await person.messages_after(token, sent) if m.actor_id == TALK_USER]
    finally:
        await person.aclose()
    assert replies, "uriel did not answer the DM within 3 minutes"
