from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]
    kind: Literal["human", "service"]
    # The OIDC subject. uriel-tools matches it to Nextcloud accounts that sociallogin named authelia-<sub>.
    sub: str | None = None

    def mcp_meta(self) -> dict[str, Any]:
        meta = {"user": self.user_id, "groups": sorted(self.groups)}
        return meta | {"sub": self.sub} if self.sub else meta

    def to_session(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "groups": sorted(self.groups), "kind": self.kind, "sub": self.sub}

    @classmethod
    def from_session(cls, data: dict[str, Any]) -> "Principal":
        # Sessions from before `sub` existed lack it; they pick it up at the next sign-in.
        return cls(data["user_id"], frozenset(data["groups"]), data["kind"], data.get("sub"))
