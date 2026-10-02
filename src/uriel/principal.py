from dataclasses import dataclass
from typing import Any, Literal

INTERNAL_GROUP = "uriel-internal"


@dataclass(frozen=True)
class Principal:
    user_id: str
    groups: frozenset[str]
    kind: Literal["human", "service"]
    # The OIDC subject. uriel-tools matches it to Nextcloud accounts that sociallogin named authelia-<sub>.
    sub: str | None = None

    def __post_init__(self):
        # uriel-internal unlocks the runner's hidden tools (claim_due_runs, finish_run): no person may
        # hold it, whatever lldap, a token or an old session says.
        if self.kind == "human" and INTERNAL_GROUP in self.groups:
            object.__setattr__(self, "groups", self.groups - {INTERNAL_GROUP})

    def mcp_meta(self) -> dict[str, Any]:
        meta = {"user": self.user_id, "groups": sorted(self.groups)}
        return meta | {"sub": self.sub} if self.sub else meta

    def to_session(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "groups": sorted(self.groups), "kind": self.kind, "sub": self.sub}

    @classmethod
    def from_session(cls, data: dict[str, Any]) -> "Principal":
        # Sessions from before `sub` existed lack it; they pick it up at the next sign-in.
        return cls(data["user_id"], frozenset(data["groups"]), data["kind"], data.get("sub"))
