import os

import pytest

from uriel.gateway.directory import LdapGroupDirectory

pytestmark = pytest.mark.live


async def test_real_lldap_knows_the_family():
    # lldap is ClusterIP-only: port-forward it first (docs/development.md, "Talk and scheduled runs").
    if not os.environ.get("URIEL_LDAP_URL"):
        pytest.skip("URIEL_LDAP_URL not set")
    d = LdapGroupDirectory(
        os.environ["URIEL_LDAP_URL"],
        os.environ["URIEL_LDAP_BIND_DN"],
        os.environ["URIEL_LDAP_PASSWORD"],
        os.environ["URIEL_LDAP_BASE_DN"],
    )
    user = os.environ.get("URIEL_LIVE_LDAP_USER", "anthony-headband")
    assert {"admins", "family"} <= await d.groups_of(user)
    assert await d.groups_of("authelia-not-a-person") is None
