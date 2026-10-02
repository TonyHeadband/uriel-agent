from uriel.gateway.auth import principal_from_claims
from uriel.principal import INTERNAL_GROUP, Principal


def test_no_person_holds_the_internal_group():
    p = Principal("dad", frozenset({"family", INTERNAL_GROUP}), "human")
    assert p.groups == frozenset({"family"})
    assert Principal.from_session({**p.to_session(), "groups": ["family", INTERNAL_GROUP]}).groups == {
        "family"
    }
    claims = {"preferred_username": "dad", "groups": ["family", INTERNAL_GROUP]}
    assert principal_from_claims(claims).groups == frozenset({"family"})


def test_the_gateways_own_service_principal_keeps_it():
    assert INTERNAL_GROUP in Principal("uriel-gateway", frozenset({INTERNAL_GROUP}), "service").groups
