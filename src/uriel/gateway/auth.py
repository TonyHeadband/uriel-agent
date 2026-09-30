import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jwt
import yaml

from uriel.principal import Principal


class AuthError(Exception):
    pass


def principal_from_claims(claims: Mapping[str, Any]) -> Principal:
    groups = claims.get("groups")
    if not isinstance(groups, list) or not groups:
        # Authelia 4.39 only puts groups in the ID token when the client has a claims_policy for it.
        raise AuthError("the identity token has no groups claim; check the Authelia claims_policy for uriel")
    user = claims.get("preferred_username") or claims.get("sub")
    if not user:
        raise AuthError("the identity token has no user")
    sub = claims.get("sub")
    return Principal(str(user), frozenset(str(g) for g in groups), "human", str(sub) if sub else None)


@dataclass(frozen=True)
class _ServiceKey:
    sha256: str
    principal: Principal


class ServiceKeys:
    """API keys for agents and devices. Only SHA-256 digests are stored."""

    def __init__(self, keys: list[_ServiceKey]):
        self._keys = keys

    @classmethod
    def empty(cls) -> "ServiceKeys":
        return cls([])

    @classmethod
    def from_file(cls, path: Path) -> "ServiceKeys":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            [
                _ServiceKey(k["sha256"].lower(), Principal(k["user"], frozenset(k["groups"]), "service"))
                for k in raw.get("keys", [])
            ]
        )

    def lookup(self, presented: str) -> Principal | None:
        digest = hashlib.sha256(presented.encode()).hexdigest()
        found = None
        for key in self._keys:  # no early exit, so timing doesn't reveal which key matched
            if hmac.compare_digest(digest, key.sha256):
                found = key.principal
        return found


class JwtVerifier:
    def __init__(self, issuer: str, audience: str, jwks_client: Any):
        self._issuer = issuer
        self._audience = audience
        self._jwks = jwks_client

    @classmethod
    def for_issuer(cls, issuer: str, audience: str) -> "JwtVerifier":
        return cls(issuer, audience, jwt.PyJWKClient(f"{issuer}/jwks.json", cache_keys=True, lifespan=300))

    def verify(self, token: str) -> Principal:
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=30,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        return principal_from_claims(claims)
