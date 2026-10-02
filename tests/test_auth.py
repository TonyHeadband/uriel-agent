import hashlib
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from uriel.gateway.auth import AuthError, JwtVerifier, ServiceKeys, principal_from_claims

ISS, AUD = "https://auth.example", "uriel"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeJwks:
    def get_signing_key_from_jwt(self, token):
        return type("K", (), {"key": KEY.public_key()})()


def token(**overrides):
    claims = {
        "iss": ISS,
        "aud": AUD,
        "sub": "abc",
        "preferred_username": "dad",
        "groups": ["admins", "family"],
        "exp": int(time.time()) + 300,
    } | overrides
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, KEY, algorithm="RS256")


def test_valid_token_gives_principal():
    p = JwtVerifier(ISS, AUD, FakeJwks()).verify(token())
    assert (p.user_id, p.groups, p.kind) == ("dad", frozenset({"admins", "family"}), "human")


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": int(time.time()) - 600},
        {"aud": "other"},
        {"iss": "https://evil"},
        {"groups": None},
        {"groups": []},
    ],
)
def test_bad_tokens_are_rejected(overrides):
    with pytest.raises(AuthError):
        JwtVerifier(ISS, AUD, FakeJwks()).verify(token(**overrides))


def test_token_signed_by_other_key_is_rejected():
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    claims = {
        "iss": ISS,
        "aud": AUD,
        "sub": "x",
        "groups": ["admins"],
        "exp": int(time.time()) + 60,
    }
    forged = jwt.encode(claims, other, algorithm="RS256")
    with pytest.raises(AuthError):
        JwtVerifier(ISS, AUD, FakeJwks()).verify(forged)


def test_claims_without_groups_are_refused():
    with pytest.raises(AuthError, match="groups"):
        principal_from_claims({"preferred_username": "dad"})


def test_claims_keep_the_subject_for_nextcloud_matching():
    p = principal_from_claims({"preferred_username": "lil-b", "sub": "19e1429e", "groups": ["family"]})
    assert (p.user_id, p.sub) == ("lil-b", "19e1429e")


def test_service_keys_match_by_hash(tmp_path):
    f = tmp_path / "keys.yaml"
    digest = hashlib.sha256(b"kitchen-secret").hexdigest()
    f.write_text(f"keys:\n  - sha256: {digest}\n    user: kitchen\n    groups: [family]\n")
    keys = ServiceKeys.from_file(f)
    p = keys.lookup("kitchen-secret")
    assert (p.user_id, p.groups, p.kind) == ("kitchen", frozenset({"family"}), "service")
    assert keys.lookup("nope") is None
    assert ServiceKeys.empty().lookup("kitchen-secret") is None


def test_any_accepted_audience_is_enough():
    verifier = JwtVerifier(ISS, [AUD, "uriel-companion"], FakeJwks())
    assert verifier.verify(token(aud="uriel-companion")).user_id == "dad"
    assert verifier.verify(token()).user_id == "dad"
    with pytest.raises(AuthError):
        verifier.verify(token(aud="other"))
