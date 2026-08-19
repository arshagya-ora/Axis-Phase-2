"""Strict runtime configuration for the AXIS control plane."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class ModelMode(StrEnum):
    MOCK = "mock"
    OCI_OPENAI = "oci_openai"


class OciAuthMode(StrEnum):
    """Explicit OCI authentication strategy; credentials are never auto-detected."""

    API_KEY = "api_key"
    USER_PRINCIPAL = "user_principal"


OCI_API_KEY_REGION_KEYS: dict[str, str] = {
    "ap-hyderabad-1": "hyd",
    "ap-osaka-1": "kix",
    "eu-frankfurt-1": "fra",
    "us-ashburn-1": "iad",
    "us-chicago-1": "ord",
    "us-phoenix-1": "phx",
}
_OCI_INFERENCE_HOST_PREFIX = "inference.generativeai."
_OCI_INFERENCE_HOST_SUFFIX = ".oci.oraclecloud.com"
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,254}$")
_OCI_PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

DEFAULT_OCI_BASE_URL = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1"
DEFAULT_PLANNER_MODEL = "xai.grok-4.3"
DEFAULT_NAVIGATOR_MODEL = "xai.grok-4.20-0309-non-reasoning"


class PlaywrightBrowser(StrEnum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class PlaywrightSettings(BaseSettings):
    """Validated settings for AXIS-owned Playwright browser sessions."""

    model_config = SettingsConfigDict(
        env_prefix="AXIS_PLAYWRIGHT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    browser: PlaywrightBrowser = PlaywrightBrowser.CHROMIUM
    headless: bool = True
    user_data_dir: Path | None = None
    downloads_path: Path = Path(".axis-data/downloads")
    default_timeout_ms: int = Field(default=15_000, ge=1_000, le=120_000)
    navigation_timeout_ms: int = Field(default=30_000, ge=1_000, le=120_000)
    viewport_width: int = Field(default=1440, ge=320, le=7680)
    viewport_height: int = Field(default=900, ge=240, le=4320)
    locale: str = Field(default="en-US", min_length=2, max_length=32)
    timezone_id: str = Field(default="UTC", min_length=1, max_length=64)
    slow_mo_ms: int = Field(default=0, ge=0, le=5_000)
    max_observation_text_chars: int = Field(default=20_000, ge=1_000, le=100_000)
    max_cached_content_chars: int = Field(default=100_000, ge=1_000, le=1_000_000)
    search_base_url: str = "https://www.google.com/search?q="

    @field_validator("search_base_url")
    @classmethod
    def require_https_search_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("search_base_url must be a clean absolute HTTPS query prefix") from exc
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.query
            or parsed.fragment
            or not value.endswith("=")
        ):
            raise ValueError("search_base_url must be a clean absolute HTTPS query prefix")
        return value

    @model_validator(mode="after")
    def validate_storage_paths(self) -> Self:
        if self.user_data_dir is not None and self.user_data_dir == self.downloads_path:
            raise ValueError("user_data_dir and downloads_path must be different")
        return self


class AxisSettings(BaseSettings):
    """Environment-backed settings with strict constructor and dotenv validation."""

    model_config = SettingsConfigDict(
        env_prefix="AXIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Environment = Environment.DEVELOPMENT
    model_mode: ModelMode = ModelMode.MOCK
    database_path: Path = Path(".axis-data/axis.db")
    firewall_policy_path: Path | None = None
    server_host: str = "127.0.0.1"
    server_port: int = Field(default=7770, ge=1024, le=65535)
    log_level: str = "INFO"

    max_steps: int = Field(default=100, ge=1, le=100)
    max_actions_per_step: int = Field(default=10, ge=1, le=10)
    max_failures: int = Field(default=3, ge=1, le=10)
    planning_interval: int = Field(default=3, ge=1, le=100)
    action_timeout_seconds: int = Field(default=30, ge=1, le=120)
    model_timeout_seconds: int = Field(default=120, ge=1, le=600)
    workflow_timeout_seconds: int = Field(default=1_800, ge=1, le=86_400)

    oci_base_url: str = DEFAULT_OCI_BASE_URL
    oci_project_ocid: str | None = None
    oci_auth_mode: OciAuthMode = OciAuthMode.API_KEY
    oci_genai_api_key: SecretStr | None = None
    oci_config_file: Path = Path("~/.oci/config")
    oci_profile: str = "DEFAULT"
    planner_model: str = DEFAULT_PLANNER_MODEL
    navigator_model: str = DEFAULT_NAVIGATOR_MODEL

    @field_validator(
        "oci_base_url",
        "oci_project_ocid",
        "planner_model",
        "navigator_model",
        "oci_profile",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        return value

    @field_validator("oci_config_file", mode="before")
    @classmethod
    def require_nonempty_oci_config_path(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                raise ValueError("oci_config_file must not be empty")
            return normalized
        return value

    @field_validator("oci_genai_api_key", mode="before")
    @classmethod
    def normalize_optional_secret(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("oci_genai_api_key")
    @classmethod
    def validate_oci_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        secret = value.get_secret_value()
        if secret != secret.strip() or len(secret) < 20:
            raise ValueError("oci_genai_api_key must be a trimmed runtime secret")
        return value

    @field_validator("oci_base_url")
    @classmethod
    def validate_oci_base_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("oci_base_url must be a valid OCI HTTPS endpoint") from exc
        hostname = parsed.hostname
        if (
            parsed.scheme != "https"
            or hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or port is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/openai/v1"
            or not hostname.startswith(_OCI_INFERENCE_HOST_PREFIX)
            or not hostname.endswith(_OCI_INFERENCE_HOST_SUFFIX)
        ):
            raise ValueError(
                "oci_base_url must use the OCI inference HTTPS origin and /openai/v1 path"
            )
        region = hostname[len(_OCI_INFERENCE_HOST_PREFIX) : -len(_OCI_INFERENCE_HOST_SUFFIX)]
        if region not in OCI_API_KEY_REGION_KEYS:
            raise ValueError("oci_base_url region is not supported by AXIS")
        return f"https://{hostname}/openai/v1"

    @field_validator("oci_project_ocid")
    @classmethod
    def validate_oci_project_ocid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if re.fullmatch(r"ocid1\.generativeaiproject\.oc1\.[a-z0-9-]+\.[a-z0-9]+", value) is None:
            raise ValueError("oci_project_ocid must be an OC1 Generative AI project OCID")
        return value

    @field_validator("planner_model", "navigator_model")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        if _MODEL_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("model IDs must contain only safe OCI model identifier characters")
        return value

    @field_validator("oci_profile")
    @classmethod
    def validate_oci_profile(cls, value: str) -> str:
        if _OCI_PROFILE_PATTERN.fullmatch(value) is None:
            raise ValueError("oci_profile contains unsupported characters")
        return value

    @field_validator("server_host")
    @classmethod
    def require_loopback_server(cls, value: str) -> str:
        if value not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("server_host must be loopback")
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("invalid log level")
        return normalized

    @model_validator(mode="after")
    def validate_runtime(self) -> Self:
        if self.planning_interval > self.max_steps:
            raise ValueError("planning_interval cannot exceed max_steps")
        if self.model_mode is ModelMode.OCI_OPENAI:
            required: dict[str, object | None] = {
                "AXIS_OCI_PROJECT_OCID": self.oci_project_ocid,
            }
            if self.oci_auth_mode is OciAuthMode.API_KEY:
                required["AXIS_OCI_GENAI_API_KEY"] = self.oci_genai_api_key
            elif self.oci_genai_api_key is not None:
                raise ValueError(
                    "AXIS_OCI_GENAI_API_KEY must not be set with AXIS_OCI_AUTH_MODE=user_principal"
                )
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(
                    "OCI model mode requires runtime configuration: " + ", ".join(missing)
                )
            self._validate_oci_region_alignment()
        if self.environment is Environment.PRODUCTION and self.firewall_policy_path is None:
            raise ValueError("production requires a firewall policy file")
        if not str(self.database_path).strip():
            raise ValueError("database_path must not be empty")
        return self

    def _validate_oci_region_alignment(self) -> None:
        if self.oci_project_ocid is None:
            return
        hostname = urlsplit(self.oci_base_url).hostname
        if hostname is None:
            raise ValueError("oci_base_url must contain a hostname")
        region = hostname[len(_OCI_INFERENCE_HOST_PREFIX) : -len(_OCI_INFERENCE_HOST_SUFFIX)]
        ocid_parts = self.oci_project_ocid.split(".")
        project_region = ocid_parts[3]
        accepted_project_regions = {region, OCI_API_KEY_REGION_KEYS[region]}
        if project_region not in accepted_project_regions:
            raise ValueError("OCI endpoint region and project OCID region must match")

    @property
    def oci_region(self) -> str:
        hostname = urlsplit(self.oci_base_url).hostname
        if hostname is None:
            raise ValueError("oci_base_url must contain a hostname")
        return hostname[len(_OCI_INFERENCE_HOST_PREFIX) : -len(_OCI_INFERENCE_HOST_SUFFIX)]

    def public_snapshot(self) -> dict[str, object]:
        """Return a diagnostic snapshot that can never include secret values."""

        return {
            "environment": self.environment.value,
            "modelMode": self.model_mode.value,
            "databasePath": str(self.database_path),
            "firewallPolicyPath": (
                str(self.firewall_policy_path) if self.firewall_policy_path else None
            ),
            "serverHost": self.server_host,
            "serverPort": self.server_port,
            "logLevel": self.log_level,
            "maxSteps": self.max_steps,
            "maxActionsPerStep": self.max_actions_per_step,
            "maxFailures": self.max_failures,
            "planningInterval": self.planning_interval,
            "actionTimeoutSeconds": self.action_timeout_seconds,
            "modelTimeoutSeconds": self.model_timeout_seconds,
            "workflowTimeoutSeconds": self.workflow_timeout_seconds,
            "ociConfigured": self.model_mode is ModelMode.OCI_OPENAI,
            "ociAuthMode": self.oci_auth_mode.value,
            "ociRegion": self.oci_region,
            "plannerModel": self.planner_model,
            "navigatorModel": self.navigator_model,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.public_snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_settings(**overrides: object) -> AxisSettings:
    return AxisSettings(**overrides)  # type: ignore[arg-type]
