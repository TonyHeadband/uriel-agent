from pathlib import Path

import pytest
from pydantic import ValidationError

from uriel.config import DeciderSpec, GatewaySettings, ModelsConfig, load_yaml, models_outside_the_house
from uriel.principal import Principal

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_model_configs_are_valid():
    for name in ("models.yaml", "models.dev.yaml"):
        cfg = load_yaml(ModelsConfig, ROOT / "config" / name)
        assert cfg.role("interactive").model == "qwen3:8b"
        assert cfg.decider("route").adapter == "llm"
        # Roles uriel-tools reads from the same file.
        assert {"embedding", "ocr"} <= cfg.roles.keys()
        assert cfg.role("tools").params["reasoning_effort"] == "low"


def test_tools_role_falls_back_to_the_chat_model():
    cfg = load_yaml(ModelsConfig, ROOT / "tests" / "e2e" / "models.e2e.yaml")
    assert cfg.role("tools") == cfg.role("interactive")


def test_unknown_model_reference_is_rejected():
    raw = {
        "models": {"a": {"provider": "openai_compat", "base_url": "http://x/v1", "model": "m"}},
        "roles": {"interactive": "missing", "background": "a"},
        "deciders": {"route": {"adapter": "llm", "model": "a"}},
    }
    with pytest.raises(ValidationError, match="unknown model reference"):
        ModelsConfig.model_validate(raw)


def test_gateway_roles_are_required():
    raw = {
        "models": {"a": {"provider": "openai_compat", "base_url": "http://x/v1", "model": "m"}},
        "roles": {"interactive": "a", "embedding": "a"},
        "deciders": {"route": {"adapter": "llm", "model": "a"}},
    }
    with pytest.raises(ValidationError, match="missing role"):
        ModelsConfig.model_validate(raw)


def test_decider_adapter_must_match_its_model_provider():
    raw = {
        "models": {"a": {"provider": "openai_compat", "base_url": "http://x/v1", "model": "m"}},
        "roles": {"interactive": "a", "background": "a"},
        "deciders": {"route": {"adapter": "systemone", "model": "a"}},
    }
    with pytest.raises(ValidationError, match="systemone"):
        ModelsConfig.model_validate(raw)


def test_a_systemone_model_cannot_fill_a_chat_role():
    raw = {
        "models": {"t": {"provider": "systemone", "base_url": "http://x/v1", "model": "tev1:0.8b"}},
        "roles": {"interactive": "t", "background": "t"},
        "deciders": {"route": {"adapter": "systemone", "model": "t"}},
    }
    with pytest.raises(ValidationError, match="chat"):
        ModelsConfig.model_validate(raw)


def test_decider_coverage_is_a_probability():
    assert DeciderSpec(adapter="systemone", model="t", coverage=0.9).coverage == 0.9
    with pytest.raises(ValidationError):
        DeciderSpec(adapter="systemone", model="t", coverage=1.5)


def test_an_llm_decider_can_narrow_by_its_single_choice():
    raw = {
        "models": {"a": {"provider": "openai_compat", "base_url": "http://x/v1", "model": "m"}},
        "roles": {"interactive": "a", "background": "a"},
        "deciders": {"route": {"adapter": "llm", "model": "a", "coverage": 0.9}},
    }
    assert ModelsConfig.model_validate(raw).decider("route").coverage == 0.9


def test_principal_session_roundtrip_and_meta():
    p = Principal("mom", frozenset({"family", "admins"}), "human")
    assert Principal.from_session(p.to_session()) == p
    assert p.mcp_meta() == {"user": "mom", "groups": ["admins", "family"]}
    with_sub = Principal("lil-b", frozenset({"family"}), "human", "19e1429e")
    assert Principal.from_session(with_sub.to_session()) == with_sub
    assert with_sub.mcp_meta() == {"user": "lil-b", "groups": ["family"], "sub": "19e1429e"}
    old_session = {"user_id": "mom", "groups": ["family"], "kind": "human"}
    assert Principal.from_session(old_session).sub is None


def test_gateway_settings_read_env(monkeypatch):
    for k, v in {
        "URIEL_DATABASE_URL": "postgresql://u:p@db/uriel",
        "URIEL_MCP_URL": "http://mcp:8001/mcp",
        "URIEL_MCP_API_KEY": "k",
        "URIEL_OIDC_ISSUER": "https://auth.example",
        "URIEL_OIDC_CLIENT_ID": "uriel",
        "URIEL_OIDC_CLIENT_SECRET": "s",
        "URIEL_SESSION_SECRET": "x" * 32,
        "URIEL_PUBLIC_URL": "https://uriel.example",
        "URIEL_OIDC_EXTRA_AUDIENCES": '["uriel-companion"]',
        "URIEL_CORS_ORIGINS": '["tauri://localhost"]',
    }.items():
        monkeypatch.setenv(k, v)
    s = GatewaySettings()
    assert s.cookie_secure is True
    assert s.max_message_chars == 8000
    assert s.oidc_extra_audiences == ["uriel-companion"]
    assert s.cors_origins == ["tauri://localhost"]


def test_companion_settings_default_to_empty(monkeypatch):
    for k, v in {
        "URIEL_DATABASE_URL": "postgresql://u:p@db/uriel",
        "URIEL_MCP_URL": "http://mcp:8001/mcp",
        "URIEL_MCP_API_KEY": "k",
        "URIEL_OIDC_ISSUER": "https://auth.example",
        "URIEL_OIDC_CLIENT_ID": "uriel",
        "URIEL_OIDC_CLIENT_SECRET": "s",
        "URIEL_SESSION_SECRET": "x" * 32,
        "URIEL_PUBLIC_URL": "https://uriel.example",
    }.items():
        monkeypatch.setenv(k, v)
    s = GatewaySettings()
    assert (s.oidc_extra_audiences, s.cors_origins) == ([], [])


def _models(base_url: str, **spec) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "models": {"m": {"provider": "openai_compat", "base_url": base_url, "model": "x", **spec}},
            "roles": {"interactive": "m", "background": "m"},
            "deciders": {"route": {"adapter": "llm", "model": "m"}},
        }
    )


def test_api_key_can_come_from_the_environment(monkeypatch):
    spec = _models("http://x/v1", api_key_env="EVAL_KEY").models["m"]
    monkeypatch.setenv("EVAL_KEY", "s3cret")
    assert spec.key() == "s3cret"
    monkeypatch.delenv("EVAL_KEY")
    with pytest.raises(ValueError, match="EVAL_KEY"):
        spec.key()
    assert _models("http://x/v1").models["m"].key() == "unused"


def fake_resolver(table):
    def resolve(host, port, *a, **kw):
        if host not in table:
            raise OSError(f"cannot resolve {host}")
        return [(2, 1, 6, "", (ip, port or 0)) for ip in table[host]]

    return resolve


@pytest.mark.parametrize(
    ("url", "outside"),
    [
        ("http://192.168.1.10:31434/v1", False),
        ("http://localhost:11434/v1", False),
        ("http://stub-llm:9000/v1", False),  # a compose service name
        ("http://ollama.ai.svc.cluster.local:11434/v1", False),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", True),
        ("http://mixed:1/v1", True),  # one public address is enough
        ("http://nowhere:1/v1", True),  # can't tell where it is, so it isn't allowed
    ],
)
def test_models_outside_the_house_are_found(url, outside):
    resolve = fake_resolver(
        {
            "localhost": ["127.0.0.1"],
            "192.168.1.10": ["192.168.1.10"],
            "stub-llm": ["172.25.0.4"],
            "ollama.ai.svc.cluster.local": ["10.96.0.10"],
            "generativelanguage.googleapis.com": ["142.250.184.10"],
            "mixed": ["10.0.0.2", "8.8.8.8"],
        }
    )
    assert (models_outside_the_house(_models(url), resolve) == ["m"]) == outside


def test_shipped_model_configs_stay_in_the_house():
    for name in ("models.yaml", "models.dev.yaml"):
        cfg = load_yaml(ModelsConfig, ROOT / "config" / name)
        # Hosted models need a key; the house's Ollama doesn't. Hosts are checked when the gateway starts.
        assert not [m for m in cfg.models.values() if m.api_key_env], name


BASE = dict(
    database_url="postgresql://unused",
    mcp_url="http://unused",
    mcp_api_key="k",
    oidc_issuer="https://auth.example",
    oidc_client_id="uriel",
    oidc_client_secret="s",
    session_secret="s" * 32,
    public_url="http://testserver",
)
COWORKER = dict(
    nc_url="https://cloud.example",
    nc_app_password="p",
    ldap_url="ldap://lldap:3890",
    ldap_bind_dn="uid=uriel-gateway,ou=people,dc=example,dc=com",
    ldap_password="p",
    ldap_base_dn="dc=example,dc=com",
)


def test_talk_and_runner_are_off_by_default_and_need_nothing_more():
    s = GatewaySettings(**BASE)
    assert (s.talk_enabled, s.runner_enabled) == (False, False)
    assert (s.talk_poll_seconds, s.talk_history_turns, s.runner_seconds) == (3.0, 20, 30.0)


@pytest.mark.parametrize("flag", ["talk_enabled", "runner_enabled"])
def test_talk_or_runner_need_nextcloud_and_lldap(flag):
    with pytest.raises(ValidationError, match="URIEL_NC_URL.*URIEL_LDAP_BASE_DN"):
        GatewaySettings(**BASE, **{flag: True})
    # Compose passes an unset variable as an empty string.
    with pytest.raises(ValidationError, match="URIEL_LDAP_PASSWORD"):
        GatewaySettings(**BASE, **(COWORKER | {"ldap_password": ""}), **{flag: True})
    assert getattr(GatewaySettings(**BASE, **COWORKER, **{flag: True}), flag)


def test_member_groups_default_to_family_and_read_env(monkeypatch):
    assert GatewaySettings(**BASE).member_groups == ["family"]
    monkeypatch.setenv("URIEL_MEMBER_GROUPS", '["family", "grandparents"]')
    assert GatewaySettings(**BASE).member_groups == ["family", "grandparents"]
    with pytest.raises(ValidationError):
        GatewaySettings(**BASE, member_groups=[])
