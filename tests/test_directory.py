import pytest
from ldap3 import MOCK_SYNC, Connection, Server
from ldap3.core.exceptions import LDAPSocketOpenError

from uriel.gateway.directory import (
    CONNECTION_OPTIONS,
    GroupLookupError,
    LdapGroupDirectory,
    group_names,
    user_filter,
)

BASE = "dc=homelab-station,dc=com"


def test_group_names_come_from_memberof_under_ou_groups():
    dns = [
        "cn=family,ou=groups,dc=homelab-station,dc=com",
        "cn=admins,ou=groups,dc=homelab-station,dc=com",
        "cn=uriel-internal,ou=groups,dc=homelab-station,dc=com",
        "cn=family,ou=elsewhere,dc=homelab-station,dc=com",
        "uid=someone,ou=people,dc=homelab-station,dc=com",
    ]
    assert group_names(dns, BASE) == frozenset({"family", "admins"})


def test_the_uid_is_escaped_in_the_filter():
    assert user_filter("a*)(uid=b") == "(&(objectClass=person)(uid=a\\2a\\29\\28uid=b))"


class CountingDirectory(LdapGroupDirectory):
    """The real cache and error handling around a scripted search."""

    def __init__(self, answers, clock):
        super().__init__("ldap://unused", "uid=x", "p", BASE, ttl_s=300, clock=clock)
        self.answers, self.searches = answers, []

    def _search(self, uid):
        self.searches.append(uid)
        answer = self.answers[uid]
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_lookups_are_cached_for_five_minutes_unless_fresh():
    now = [0.0]
    d = CountingDirectory({"dad": frozenset({"family"}), "ghost": None}, lambda: now[0])
    assert await d.groups_of("dad") == frozenset({"family"})
    assert await d.groups_of("dad") == frozenset({"family"})
    assert await d.groups_of("ghost") is None
    assert await d.groups_of("ghost") is None
    assert d.searches == ["dad", "ghost"]
    now[0] = 301
    await d.groups_of("dad")
    await d.groups_of("dad", fresh=True)
    assert d.searches == ["dad", "ghost", "dad", "dad"]


async def test_an_ldap_failure_is_a_lookup_error_and_is_not_cached():
    d = CountingDirectory({"dad": LDAPSocketOpenError("unreachable")}, lambda: 0.0)
    with pytest.raises(GroupLookupError):
        await d.groups_of("dad")
    d.answers["dad"] = frozenset({"family"})
    assert await d.groups_of("dad") == frozenset({"family"})


class MockedLdap(LdapGroupDirectory):
    """The real _search against ldap3's in-memory server."""

    def __init__(self, entries):
        super().__init__("ldap://unused", "uid=x", "p", BASE)
        self.entries = entries

    def _connect(self):
        conn = Connection(
            Server("mock"),
            user=f"uid=x,ou=people,{BASE}",
            password="p",
            client_strategy=MOCK_SYNC,
            **CONNECTION_OPTIONS,
        )
        for dn, attrs in self.entries.items():
            conn.strategy.add_entry(dn, attrs)
        conn.bind()
        return conn


PEOPLE = f"ou=people,{BASE}"
DAD = {
    f"uid=x,{PEOPLE}": {"objectClass": "person", "userPassword": "p", "uid": "x"},
    f"uid=dad,{PEOPLE}": {
        "objectClass": "person",
        "uid": "dad",
        "memberOf": [f"cn=family,ou=groups,{BASE}", f"cn=uriel-internal,ou=groups,{BASE}"],
    },
}


def test_the_real_search_reads_memberof():
    assert MockedLdap(DAD)._search("dad") == frozenset({"family"})


def test_the_real_search_returns_none_for_no_such_person():
    assert MockedLdap(DAD)._search("ghost") is None


async def test_a_result_code_failure_is_a_lookup_error_and_is_not_cached():
    d = MockedLdap(DAD)
    d._base_dn = f"ou=nope,{BASE}"
    with pytest.raises(GroupLookupError):
        await d.groups_of("dad")
    assert d._cache == {}
