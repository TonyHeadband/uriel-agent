import ipaddress
import os
import socket
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSpec(BaseModel):
    # systemone: a decision model answering typed questions at {base_url}/systemone (Ollama >= 0.35).
    provider: Literal["openai_compat", "systemone"]
    base_url: str
    model: str
    api_key: str = "unused"
    api_key_env: str | None = (
        None  # read the key from this variable instead, e.g. for a hosted model in evals
    )
    timeout_s: float = 60.0
    max_retries: int = 1
    params: dict[str, Any] = Field(default_factory=dict)

    def key(self) -> str:
        if not self.api_key_env:
            return self.api_key
        try:
            return os.environ[self.api_key_env]
        except KeyError:
            raise ValueError(f"{self.api_key_env} is not set (api_key_env of model {self.model})") from None


class DeciderSpec(BaseModel):
    adapter: Literal["llm", "systemone"]
    model: str
    min_confidence: float = Field(default=0.6, ge=0, le=1)
    # Set: narrow tools by category. systemone pools categories up to this probability mass; an llm decider
    # has no probabilities, so its single choice is the pool.
    coverage: float | None = Field(default=None, gt=0, le=1)


class ModelsConfig(BaseModel):
    """models.yaml is shared by the whole deployment: uriel-tools reads its roles (embedding, ocr) from the
    same file, and the gateway ignores them."""

    models: dict[str, ModelSpec]
    roles: dict[str, str]
    deciders: dict[str, DeciderSpec]

    @model_validator(mode="after")
    def _references_exist(self) -> "ModelsConfig":
        missing_roles = sorted({"interactive", "background"} - self.roles.keys())
        if missing_roles:
            raise ValueError(f"missing role(s): {', '.join(missing_roles)}")
        refs = [*self.roles.values(), *(d.model for d in self.deciders.values())]
        missing = sorted({r for r in refs if r not in self.models})
        if missing:
            raise ValueError(f"unknown model reference(s): {', '.join(missing)}")
        not_chat = sorted(r for r in self.roles.values() if self.models[r].provider == "systemone")
        if not_chat:
            raise ValueError(f"decision model(s) can't fill a chat role: {', '.join(not_chat)}")
        for point, d in self.deciders.items():
            provider = self.models[d.model].provider
            if (d.adapter == "systemone") != (provider == "systemone"):
                raise ValueError(f"decider {point}: adapter {d.adapter} can't use a {provider} model")
        return self

    def role(self, name: Literal["interactive", "background", "tools"]) -> ModelSpec:
        # `tools` (the step that decides tool calls) is optional and falls back to the chat model.
        if name == "tools" and name not in self.roles:
            name = "interactive"
        return self.models[self.roles[name]]

    def decider(self, point: str) -> DeciderSpec:
        return self.deciders[point]


def models_outside_the_house(models: ModelsConfig, resolve=None) -> list[str]:
    """Names of models whose host resolves to any address outside private networks, or doesn't resolve.

    Family conversations and documents must never reach a hosted model. The gateway refuses to start with
    one, with no override, so a models file made for evals can't be shipped by mistake.
    """
    resolve = resolve or socket.getaddrinfo
    outside = []
    for name, spec in models.models.items():
        host = urlsplit(spec.base_url).hostname or ""
        try:
            addresses = {info[4][0] for info in resolve(host, None)}
        except OSError:
            addresses = set()
        if not addresses or not all(_in_house(a) for a in addresses):
            outside.append(name)
    return sorted(outside)


def _in_house(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])  # drop an IPv6 zone id
    return ip.is_private or ip.is_loopback or ip.is_link_local


def load_yaml[T: BaseModel](cls: type[T], path: Path | str) -> T:
    return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="URIEL_")

    database_url: str
    mcp_url: str
    mcp_api_key: str
    models_file: Path = Path("config/models.yaml")
    service_keys_file: Path | None = None
    oidc_issuer: str
    oidc_client_id: str
    oidc_client_secret: str
    # Authelia's own portal logout endpoint (Task 12 sets it after checking Authelia 4.39's path);
    # left unset, the "sign out of all family services" link is simply not shown.
    oidc_logout_url: str | None = None
    # Extra JWT audiences, for the desktop companion's own OIDC client
    # (docs/specs/2026-09-30-companion-app-design.md).
    oidc_extra_audiences: list[str] = []
    # Origins allowed to call /v1 from a webview; the companion is tauri://localhost or http://tauri.localhost.
    cors_origins: list[str] = []
    session_secret: str = Field(min_length=32)
    public_url: str
    cookie_secure: bool = True
    session_max_age_s: int = 12 * 3600
    decisions_retention_days: int = 180
    max_message_chars: int = 8000
    recursion_limit: int = 10

    # Talk channel and scheduled runs (docs/specs/2026-09-29-coworker-talk-scheduler-design.md).
    nc_url: str | None = None
    nc_app_password: str | None = None
    talk_enabled: bool = False
    talk_poll_seconds: float = Field(default=3.0, gt=0)
    talk_history_turns: int = Field(default=20, ge=1)
    ldap_url: str | None = None
    ldap_bind_dn: str | None = None
    ldap_password: str | None = None
    ldap_base_dn: str | None = None
    runner_enabled: bool = False
    runner_seconds: float = Field(default=30.0, gt=0)
    # Talk turns and scheduled runs are for people in at least one of these lldap groups, not anyone in lldap.
    member_groups: list[str] = Field(default_factory=lambda: ["family"], min_length=1)

    @model_validator(mode="after")
    def _coworker_needs_nextcloud_and_lldap(self) -> "GatewaySettings":
        if not (self.talk_enabled or self.runner_enabled):
            return self
        needed = ("nc_url", "nc_app_password", "ldap_url", "ldap_bind_dn", "ldap_password", "ldap_base_dn")
        # Compose passes an unset variable as an empty string, so empty counts as missing.
        missing = [f"URIEL_{name.upper()}" for name in needed if not getattr(self, name)]
        if missing:
            raise ValueError(f"Talk and scheduled runs need {', '.join(missing)}")
        return self
