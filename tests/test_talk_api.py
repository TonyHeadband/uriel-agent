import httpx
import pytest

from tests.fake_talk import FakeTalk
from uriel.gateway.talk_api import (
    EYES,
    Room,
    TalkClient,
    TalkError,
    TalkUnauthorized,
    TalkUnavailable,
    split_message,
)


def client_for(fake: FakeTalk, password: str = "talk-password") -> TalkClient:
    return TalkClient("http://talk.test", "uriel", password, transport=httpx.ASGITransport(app=fake.app()))


@pytest.fixture
async def talk():
    fake = FakeTalk()
    client = client_for(fake)
    yield fake, client
    await client.aclose()


async def test_rooms_are_listed_with_their_last_message(talk):
    fake, client = talk
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    fake.add_room("family", Room.GROUP, display_name="Family")
    mid = fake.say("dm-dad", "dad", "hi")
    rooms = {r.token: r for r in await client.rooms()}
    assert (rooms["dm-dad"].type, rooms["dm-dad"].name, rooms["dm-dad"].last_message_id) == (1, "dad", mid)
    assert (rooms["family"].display_name, rooms["family"].last_message_id) == ("Family", 0)


async def test_messages_after_a_cursor_come_oldest_first(talk):
    fake, client = talk
    fake.add_room("dm-dad", Room.ONE_TO_ONE, name="dad")
    ids = [fake.say("dm-dad", "dad", f"m{i}") for i in range(3)]
    messages = await client.messages_after("dm-dad", ids[0])
    assert [(m.id, m.text, m.actor_type, m.actor_id) for m in messages] == [
        (ids[1], "m1", "users", "dad"),
        (ids[2], "m2", "users", "dad"),
    ]
    assert messages[0].parameters == {} and not messages[0].system
    assert await client.messages_after("dm-dad", ids[-1]) == []


async def test_posting_replying_and_reacting(talk):
    fake, client = talk
    fake.add_room("family", Room.GROUP)
    asked = fake.say("family", "dad", "question")
    ids = await client.post_answer("family", "answer", reply_to=asked)
    assert fake.posted("family") == [
        {"token": "family", "id": ids[0], "message": "answer", "reply_to": asked}
    ]
    await client.react("family", asked)
    assert fake.reactions[("family", asked)] == {EYES}
    await client.unreact("family", asked)
    assert fake.reactions[("family", asked)] == set()


async def test_open_dm_reuses_or_creates_the_one_to_one_room(talk):
    fake, client = talk
    fake.add_room("kid-and-uriel", Room.ONE_TO_ONE, name="kid")
    assert await client.open_dm("kid") == "kid-and-uriel"
    token = await client.open_dm("dad")
    assert (fake.rooms[token]["type"], fake.rooms[token]["name"]) == (Room.ONE_TO_ONE, "dad")


@pytest.mark.parametrize(
    ("status", "error"), [(401, TalkUnauthorized), (503, TalkUnavailable), (404, TalkError)]
)
async def test_http_errors_are_typed(talk, status, error):
    fake, client = talk
    fake.fail = [status]
    with pytest.raises(error):
        await client.rooms()


async def test_a_wrong_app_password_is_unauthorized():
    client = client_for(FakeTalk(), password="wrong")
    with pytest.raises(TalkUnauthorized):
        await client.rooms()
    await client.aclose()


async def test_throttling_is_an_outage(talk):
    fake, client = talk
    fake.fail = [429]
    with pytest.raises(TalkUnavailable):
        await client.rooms()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>Sign in</html>"),
        httpx.Response(200, json={"ocs": {"meta": {"statuscode": 200}}}),
        httpx.Response(200, json={"ocs": {"meta": {"statuscode": 200}, "data": [{"nope": 1}]}}),
    ],
    ids=["not-json", "no-data", "missing-fields"],
)
async def test_a_200_that_is_not_ocs_is_an_outage(response):
    client = TalkClient("http://talk.test", "uriel", "p", transport=httpx.MockTransport(lambda r: response))
    with pytest.raises(TalkUnavailable):
        await client.rooms()
    await client.aclose()


async def test_an_ocs_error_status_is_a_talk_error():
    body = {"ocs": {"meta": {"statuscode": 403}, "data": []}}
    client = TalkClient(
        "http://talk.test",
        "uriel",
        "p",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    with pytest.raises(TalkError):
        await client.rooms()
    await client.aclose()


async def test_unreachable_nextcloud_is_unavailable():
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = TalkClient("http://talk.test", "uriel", "p", transport=httpx.MockTransport(refuse))
    with pytest.raises(TalkUnavailable):
        await client.rooms()
    await client.aclose()


def test_long_answers_split_on_paragraphs_then_lines_then_characters():
    assert split_message("short") == ["short"]
    assert split_message("   ") == []
    a, b, c = "a" * 3000, "b" * 3000, "c" * 900
    assert split_message(f"{a}\n\n{b}\n\n{c}") == [a, f"{b}\n\n{c}"]
    line = "y" * 1500
    assert split_message("\n".join([line] * 4)) == [f"{line}\n{line}", f"{line}\n{line}"]
    assert [len(p) for p in split_message("x" * 9000)] == [4000, 4000, 1000]


async def test_the_e2e_control_routes_drive_the_same_state():
    fake = FakeTalk()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fake.app()), base_url="http://t") as http:
        token = (await http.post("/_test/rooms", json={"type": 1, "name": "kid"})).json()["token"]
        await http.post("/_test/say", json={"token": token, "actor": "kid", "message": "hi"})
        assert (await http.get("/_test/rooms")).json() == [{"token": token, "type": 1, "name": "kid"}]
    assert fake.messages[token][0]["message"] == "hi"


@pytest.mark.parametrize(
    "body",
    [
        {
            "ocs": {
                "meta": {"status": "failure", "statuscode": 400, "message": "the family's secret"},
                "data": [],
            }
        },
        "<html>the family's secret</html>",
    ],
)
async def test_a_4xx_error_names_its_status_and_ocs_code_but_never_the_body(body):
    def respond(request):
        if isinstance(body, dict):
            return httpx.Response(400, json=body)
        return httpx.Response(400, text=body)

    client = TalkClient("http://talk.test", "uriel", "p", transport=httpx.MockTransport(respond))
    with pytest.raises(TalkError) as caught:
        await client.post("family", "the family's secret")
    await client.aclose()
    assert "400" in str(caught.value) and "secret" not in str(caught.value)
    if isinstance(body, dict):
        assert "OCS 400" in str(caught.value)
