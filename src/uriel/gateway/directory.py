"""Group membership from lldap, the directory Authelia signs people in from, so Talk and web agree."""

import asyncio
import time
from collections.abc import Callable, Collection, Iterable
from typing import Protocol

from ldap3 import NONE, Connection, Server
from ldap3.core.exceptions import LDAPException
from ldap3.utils.conv import escape_filter_chars
from ldap3.utils.dn import parse_dn

from uriel.principal import INTERNAL_GROUP

# raise_exceptions turns result-code failures (wrong base DN, no access) into LDAPException; otherwise
# search() returns False with an empty response, indistinguishable from "no such person".
CONNECTION_OPTIONS = {"read_only": True, "receive_timeout": 10, "raise_exceptions": True}
MEMBER_GROUPS = frozenset({"family"})  # the default for URIEL_MEMBER_GROUPS


def is_member(groups: frozenset[str] | None, member_groups: Collection[str]) -> bool:
    """Whether lldap's answer makes this person someone Uriel works for."""
    return groups is not None and not groups.isdisjoint(member_groups)


class GroupLookupError(Exception):
    """lldap couldn't be asked; not the same as "no such person"."""


class GroupDirectory(Protocol):
    async def groups_of(self, uid: str, *, fresh: bool = False) -> frozenset[str] | None:
        """The person's groups, or None when lldap has no such user."""
        ...


def user_filter(uid: str) -> str:
    return f"(&(objectClass=person)(uid={escape_filter_chars(uid)}))"


def group_names(dns: Iterable[str], base_dn: str) -> frozenset[str]:
    groups_ou = f"ou=groups,{base_dn}".lower()
    names = set()
    for dn in dns:
        rdns = parse_dn(dn)
        if not rdns or rdns[0][0].lower() != "cn":
            continue
        if ",".join(f"{attr}={value}" for attr, value, _ in rdns[1:]).lower() == groups_ou:
            names.add(rdns[0][1])
    return frozenset(names - {INTERNAL_GROUP})


class LdapGroupDirectory:
    def __init__(
        self,
        url: str,
        bind_dn: str,
        password: str,
        base_dn: str,
        *,
        ttl_s: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._url, self._bind_dn, self._password, self._base_dn = url, bind_dn, password, base_dn
        self._ttl, self._clock = ttl_s, clock
        self._cache: dict[str, tuple[float, frozenset[str] | None]] = {}

    async def groups_of(self, uid: str, *, fresh: bool = False) -> frozenset[str] | None:
        now = self._clock()
        hit = self._cache.get(uid)
        if hit is not None and not fresh and hit[0] > now:
            return hit[1]
        try:
            groups = await asyncio.to_thread(self._search, uid)
        except LDAPException as exc:
            raise GroupLookupError(f"lldap lookup of {uid} failed: {exc}") from exc
        self._cache[uid] = (now + self._ttl, groups)
        return groups

    def _connect(self) -> Connection:
        server = Server(self._url, connect_timeout=5, get_info=NONE)
        return Connection(server, self._bind_dn, self._password, auto_bind=True, **CONNECTION_OPTIONS)

    def _search(self, uid: str) -> frozenset[str] | None:
        with self._connect() as conn:
            conn.search(f"ou=people,{self._base_dn}", user_filter(uid), attributes=["memberOf"])
            entries = [e for e in conn.response or [] if e.get("type") == "searchResEntry"]
            if not entries:
                return None
            return group_names(entries[0]["attributes"].get("memberOf", []), self._base_dn)
