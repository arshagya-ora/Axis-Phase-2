from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from axis_agent.config import (
    DEFAULT_NAVIGATOR_MODEL,
    DEFAULT_OCI_BASE_URL,
    DEFAULT_PLANNER_MODEL,
    AxisSettings,
    Environment,
    ModelMode,
    OciAuthMode,
)

OCI_SETTINGS: dict[str, object] = {
    "model_mode": "oci_openai",
    "oci_base_url": ("https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1"),
    "oci_project_ocid": "ocid1.generativeaiproject.oc1.iad.exampleproject",
    "oci_genai_api_key": "unit-test-key-material",
    "planner_model": "xai.grok-planner-test",
    "navigator_model": "xai.grok-navigator-test",
}


def make_oci_settings(**overrides: object) -> AxisSettings:
    values = {**OCI_SETTINGS, **overrides}
    return AxisSettings(_env_file=None, **values)  # type: ignore[arg-type]


def test_mock_defaults_are_standalone_and_bounded() -> None:
    settings = AxisSettings(_env_file=None)
    assert settings.model_mode is ModelMode.MOCK
    assert settings.oci_auth_mode is OciAuthMode.API_KEY
    assert settings.oci_base_url == DEFAULT_OCI_BASE_URL
    assert settings.planner_model == DEFAULT_PLANNER_MODEL
    assert settings.navigator_model == DEFAULT_NAVIGATOR_MODEL
    assert settings.max_steps == 100
    assert settings.max_actions_per_step == 10
    assert settings.max_failures == 3
    assert settings.planning_interval == 3
    assert settings.oci_genai_api_key is None


def test_oci_mode_requires_all_runtime_values() -> None:
    with pytest.raises(ValidationError) as exc_info:
        AxisSettings(_env_file=None, model_mode="oci_openai")

    message = str(exc_info.value)
    assert "AXIS_OCI_PROJECT_OCID" in message
    assert "AXIS_OCI_GENAI_API_KEY" in message
    assert "AXIS_OCI_BASE_URL" not in message
    assert "AXIS_PLANNER_MODEL" not in message
    assert "AXIS_NAVIGATOR_MODEL" not in message


def test_oci_environment_variable_names_are_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIS_MODEL_MODE", "oci_openai")
    monkeypatch.setenv("AXIS_OCI_BASE_URL", str(OCI_SETTINGS["oci_base_url"]))
    monkeypatch.setenv("AXIS_OCI_PROJECT_OCID", str(OCI_SETTINGS["oci_project_ocid"]))
    monkeypatch.setenv("AXIS_OCI_GENAI_API_KEY", str(OCI_SETTINGS["oci_genai_api_key"]))
    monkeypatch.setenv("AXIS_PLANNER_MODEL", str(OCI_SETTINGS["planner_model"]))
    monkeypatch.setenv("AXIS_NAVIGATOR_MODEL", str(OCI_SETTINGS["navigator_model"]))

    settings = AxisSettings(_env_file=None)

    assert settings.model_mode is ModelMode.OCI_OPENAI
    assert settings.oci_region == "us-ashburn-1"
    assert settings.planner_model == OCI_SETTINGS["planner_model"]
    assert settings.navigator_model == OCI_SETTINGS["navigator_model"]


def test_user_principal_mode_requires_project_but_not_api_key() -> None:
    settings = AxisSettings(
        _env_file=None,
        model_mode="oci_openai",
        oci_auth_mode="user_principal",
        oci_project_ocid=OCI_SETTINGS["oci_project_ocid"],
        oci_config_file="~/.oci/config",
        oci_profile="AXIS_PRODUCTION",
    )

    assert settings.oci_auth_mode is OciAuthMode.USER_PRINCIPAL
    assert settings.oci_genai_api_key is None
    assert settings.oci_profile == "AXIS_PRODUCTION"


def test_auth_mode_never_silently_falls_back_between_credentials() -> None:
    with pytest.raises(ValidationError, match="must not be set"):
        make_oci_settings(oci_auth_mode="user_principal")

    with pytest.raises(ValidationError) as exc_info:
        AxisSettings(
            _env_file=None,
            model_mode="oci_openai",
            oci_auth_mode="api_key",
            oci_project_ocid=OCI_SETTINGS["oci_project_ocid"],
        )
    assert "AXIS_OCI_GENAI_API_KEY" in str(exc_info.value)


@pytest.mark.parametrize("profile", ["", "name with spaces", "bad/profile"])
def test_oci_profile_rejects_empty_or_unsafe_values(profile: str) -> None:
    with pytest.raises(ValidationError):
        AxisSettings(_env_file=None, oci_profile=profile)


def test_oci_profile_is_trimmed() -> None:
    settings = AxisSettings(_env_file=None, oci_profile=" AXIS_TEST ")
    assert settings.oci_profile == "AXIS_TEST"


def test_oci_config_path_rejects_empty_values() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        AxisSettings(_env_file=None, oci_config_file="  ")

    settings = AxisSettings(_env_file=None, oci_config_file=" ~/.oci/axis-config ")
    assert settings.oci_config_file == Path("~/.oci/axis-config")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1",
        "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/v1",
        "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com:443/openai/v1",
        "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1?debug=1",
        "https://example.com/openai/v1",
        "https://inference.generativeai.uk-london-1.oci.oraclecloud.com/openai/v1",
    ],
)
def test_oci_base_url_is_strict(base_url: str) -> None:
    with pytest.raises(ValidationError, match="oci_base_url"):
        make_oci_settings(oci_base_url=base_url)


def test_oci_base_url_is_canonicalized() -> None:
    settings = make_oci_settings(
        oci_base_url=("https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1/")
    )
    assert settings.oci_base_url == OCI_SETTINGS["oci_base_url"]


def test_oci_project_type_and_region_are_validated() -> None:
    with pytest.raises(ValidationError, match="Generative AI project OCID"):
        make_oci_settings(oci_project_ocid="ocid1.compartment.oc1.iad.example")
    with pytest.raises(ValidationError, match="region must match"):
        make_oci_settings(oci_project_ocid="ocid1.generativeaiproject.oc1.ord.exampleproject")


def test_oci_project_accepts_matching_region_identifier_or_region_key() -> None:
    with_region_identifier = make_oci_settings(
        oci_project_ocid="ocid1.generativeaiproject.oc1.us-ashburn-1.exampleproject"
    )
    with_region_key = make_oci_settings(
        oci_project_ocid="ocid1.generativeaiproject.oc1.iad.exampleproject"
    )

    assert with_region_identifier.oci_region == "us-ashburn-1"
    assert with_region_key.oci_region == "us-ashburn-1"


def test_runtime_model_ids_reject_unsafe_characters() -> None:
    with pytest.raises(ValidationError, match="safe OCI model identifier"):
        make_oci_settings(planner_model="xai.grok-4.3\nunsafe")
    with pytest.raises(ValidationError, match="safe OCI model identifier"):
        make_oci_settings(navigator_model="xai model")


def test_secret_never_appears_in_snapshot_or_hash_input() -> None:
    secret = "unit-test-secret-that-must-not-leak"
    settings = make_oci_settings(oci_genai_api_key=secret)
    snapshot = settings.public_snapshot()

    assert secret not in str(snapshot)
    assert secret not in settings.config_hash
    assert str(OCI_SETTINGS["oci_project_ocid"]) not in str(snapshot)
    assert snapshot["ociConfigured"] is True
    assert snapshot["ociAuthMode"] == "api_key"
    assert snapshot["ociRegion"] == "us-ashburn-1"


@pytest.mark.parametrize("secret", ["short", " padded-unit-test-secret "])
def test_oci_api_key_rejects_short_or_whitespace_padded_values(secret: str) -> None:
    with pytest.raises(ValidationError, match="trimmed runtime secret"):
        make_oci_settings(oci_genai_api_key=secret)


def test_production_requires_policy_and_loopback() -> None:
    with pytest.raises(ValidationError, match="firewall policy"):
        AxisSettings(_env_file=None, environment=Environment.PRODUCTION)
    with pytest.raises(ValidationError, match="loopback"):
        AxisSettings(_env_file=None, server_host="0.0.0.0")


def test_invalid_limits_fail() -> None:
    with pytest.raises(ValidationError):
        AxisSettings(_env_file=None, max_actions_per_step=11)
    with pytest.raises(ValidationError, match="planning_interval"):
        AxisSettings(_env_file=None, max_steps=2, planning_interval=3)
    with pytest.raises(ValidationError, match="Extra inputs"):
        AxisSettings(_env_file=None, unexpected_setting=True)
