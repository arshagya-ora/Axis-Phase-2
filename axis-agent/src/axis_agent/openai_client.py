"""Hardened direct OpenAI SDK client for OCI structured Responses calls."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from time import monotonic
from types import TracebackType
from typing import Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ValidationError

from axis_agent.config import AxisSettings, ModelMode, OciAuthMode

StructuredModelT = TypeVar("StructuredModelT", bound=BaseModel)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_SCHEMA_PARSE_ERROR_NAMES = frozenset(
    {
        "LengthFinishReasonError",
        "ResponseValidationError",
    }
)
_REFUSAL_PARSE_ERROR_NAMES = frozenset({"ContentFilterFinishReasonError"})


class MissingOpenAIDependenciesError(RuntimeError):
    """Raised when the explicitly selected live mode lacks its dependency set."""


class OpenAIClientConfigurationError(RuntimeError):
    """Raised for a safe-to-report local client configuration failure."""


class OpenAIClientStateError(RuntimeError):
    """Raised when the owned client is used outside its async lifecycle."""


class StructuredOutputIssue(StrEnum):
    REFUSAL = "refusal"
    EMPTY = "empty"
    WRONG_TYPE = "wrong_type"
    SCHEMA = "schema"


class StructuredOutputError(RuntimeError):
    """A model response that cannot be accepted as the requested typed contract."""

    def __init__(self, issue: StructuredOutputIssue) -> None:
        self.issue = issue
        super().__init__(f"structured output rejected: {issue.value}")


class ProviderErrorCategory(StrEnum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TRANSPORT = "transport"
    PROVIDER = "provider"


class ProviderCallError(RuntimeError):
    """Sanitized provider failure that never includes response bodies or credentials."""

    def __init__(
        self,
        category: ProviderErrorCategory,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.category = category
        self.status_code = status_code
        self.request_id = request_id
        self.latency_ms = latency_ms
        details = [f"category={category.value}"]
        if status_code is not None:
            details.append(f"status={status_code}")
        if request_id is not None:
            details.append(f"request_id={request_id}")
        if latency_ms is not None:
            details.append(f"latency_ms={latency_ms}")
        super().__init__("provider call failed: " + " ".join(details))


@runtime_checkable
class StructuredOutputClient(Protocol):
    """Minimal interface shared by the production client and offline test doubles."""

    async def parse_structured(
        self,
        *,
        model: str,
        input: str,
        output_type: type[StructuredModelT],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredModelT:
        """Return only an instance validated against ``output_type``."""


class _AsyncHttpClient(Protocol):
    async def aclose(self) -> None:
        """Close the owned HTTP transport."""


class _ResponsesResource(Protocol):
    def parse(self, **kwargs: object) -> Awaitable[object]:
        """Call the SDK's asynchronous structured Responses parser."""


class _AsyncOpenAIClient(Protocol):
    responses: _ResponsesResource

    async def close(self) -> None:
        """Close the SDK client and its custom HTTP transport."""


@dataclass(frozen=True, slots=True)
class OpenAISdkBindings:
    """Injectable constructors keep imports lazy and all unit tests offline."""

    create_http_client: Callable[..., object]
    create_openai_client: Callable[..., object]
    create_http_timeout: Callable[..., object]
    create_http_limits: Callable[..., object]
    create_user_principal_auth: Callable[..., object] | None = None


SdkLoader = Callable[[OciAuthMode], OpenAISdkBindings]


class OpenAIClientRuntime(StructuredOutputClient):
    """Own one direct ``AsyncOpenAI`` client for an AXIS application lifetime."""

    def __init__(
        self,
        settings: AxisSettings,
        *,
        sdk_loader: SdkLoader | None = None,
    ) -> None:
        if settings.model_mode is not ModelMode.OCI_OPENAI:
            raise OpenAIClientConfigurationError(
                "direct OpenAI runtime requires model_mode=oci_openai"
            )
        if settings.oci_project_ocid is None:
            raise OpenAIClientConfigurationError("OCI project configuration is incomplete")
        self._settings = settings
        self._sdk_loader = sdk_loader or load_openai_sdk_bindings
        self._client: _AsyncOpenAIClient | None = None

    async def __aenter__(self) -> OpenAIClientRuntime:
        if self._client is not None:
            raise OpenAIClientStateError("direct OpenAI runtime is already open")

        bindings = self._sdk_loader(self._settings.oci_auth_mode)
        auth: object | None = None
        api_key: str
        if self._settings.oci_auth_mode is OciAuthMode.API_KEY:
            secret = self._settings.oci_genai_api_key
            if secret is None:
                raise OpenAIClientConfigurationError("OCI API-key configuration is incomplete")
            api_key = secret.get_secret_value()
        else:
            api_key = "not-used"
            auth_factory = bindings.create_user_principal_auth
            if auth_factory is None:
                raise MissingOpenAIDependenciesError(
                    "OCI user-principal authentication dependency is unavailable; "
                    "install axis-agent[agent]"
                )
            config_file = _resolve_user_principal_config(self._settings.oci_config_file)
            try:
                auth = auth_factory(
                    config_file=str(config_file),
                    profile_name=self._settings.oci_profile,
                )
            except Exception:
                raise ProviderCallError(ProviderErrorCategory.AUTH) from None

        timeout = bindings.create_http_timeout(
            connect=10.0,
            read=float(self._settings.model_timeout_seconds),
            write=30.0,
            pool=5.0,
        )
        limits = bindings.create_http_limits(
            max_connections=10,
            max_keepalive_connections=5,
        )
        http_kwargs: dict[str, object] = {
            "follow_redirects": False,
            "trust_env": False,
            "timeout": timeout,
            "limits": limits,
        }
        if auth is not None:
            http_kwargs["auth"] = auth

        try:
            http_client = cast(_AsyncHttpClient, bindings.create_http_client(**http_kwargs))
        except Exception:
            raise ProviderCallError(ProviderErrorCategory.TRANSPORT) from None

        try:
            client = cast(
                _AsyncOpenAIClient,
                bindings.create_openai_client(
                    base_url=self._settings.oci_base_url,
                    api_key=api_key,
                    project=self._settings.oci_project_ocid,
                    http_client=http_client,
                    max_retries=2,
                ),
            )
        except Exception:
            await http_client.aclose()
            raise ProviderCallError(ProviderErrorCategory.TRANSPORT) from None

        self._client = client
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.close()
            except Exception:
                raise ProviderCallError(ProviderErrorCategory.TRANSPORT) from None

    async def parse_structured(
        self,
        *,
        model: str,
        input: str,
        output_type: type[StructuredModelT],
        instructions: str | None = None,
        max_output_tokens: int | None = None,
    ) -> StructuredModelT:
        client = self._client
        if client is None:
            raise OpenAIClientStateError("direct OpenAI runtime is not open")
        if not model.strip():
            raise ValueError("model must not be empty")
        if not isinstance(output_type, type) or not issubclass(output_type, BaseModel):
            raise TypeError("output_type must be a Pydantic BaseModel class")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")

        request: dict[str, object] = {
            "model": model,
            "input": input,
            "text_format": output_type,
            "store": False,
            "timeout": float(self._settings.model_timeout_seconds),
        }
        if instructions is not None:
            request["instructions"] = instructions
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        started_at = monotonic()
        try:
            response = await client.responses.parse(**request)
        except StructuredOutputError:
            raise
        except ProviderCallError:
            raise
        except ValidationError:
            raise StructuredOutputError(StructuredOutputIssue.SCHEMA) from None
        except Exception as exc:
            class_name = type(exc).__name__
            if class_name in _REFUSAL_PARSE_ERROR_NAMES:
                raise StructuredOutputError(StructuredOutputIssue.REFUSAL) from None
            if class_name in _SCHEMA_PARSE_ERROR_NAMES:
                raise StructuredOutputError(StructuredOutputIssue.SCHEMA) from None
            latency_ms = max(0, round((monotonic() - started_at) * 1_000))
            raise _sanitize_provider_error(exc, latency_ms=latency_ms) from None

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            issue = (
                StructuredOutputIssue.REFUSAL
                if _response_contains_refusal(response)
                else StructuredOutputIssue.EMPTY
            )
            raise StructuredOutputError(issue)
        if not isinstance(parsed, output_type):
            raise StructuredOutputError(StructuredOutputIssue.WRONG_TYPE)
        return parsed


def load_openai_sdk_bindings(auth_mode: OciAuthMode) -> OpenAISdkBindings:
    """Import only the dependencies required by the explicitly selected auth mode."""

    try:
        httpx_module = import_module("httpx")
        openai_module = import_module("openai")
        create_user_principal_auth: Callable[..., object] | None = None
        if auth_mode is OciAuthMode.USER_PRINCIPAL:
            auth_module = import_module("oci_genai_auth")
            create_user_principal_auth = cast(
                Callable[..., object],
                auth_module.OciUserPrincipalAuth,
            )
        return OpenAISdkBindings(
            create_http_client=cast(Callable[..., object], httpx_module.AsyncClient),
            create_openai_client=cast(
                Callable[..., object],
                openai_module.AsyncOpenAI,
            ),
            create_http_timeout=cast(Callable[..., object], httpx_module.Timeout),
            create_http_limits=cast(Callable[..., object], httpx_module.Limits),
            create_user_principal_auth=create_user_principal_auth,
        )
    except (AttributeError, ModuleNotFoundError):
        raise MissingOpenAIDependenciesError(
            "direct OpenAI dependencies are unavailable; install axis-agent[agent]"
        ) from None


def _resolve_user_principal_config(config_file: Path) -> Path:
    try:
        resolved = config_file.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise OpenAIClientConfigurationError(
            "OCI user-principal config file is unavailable"
        ) from None
    if not resolved.is_file():
        raise OpenAIClientConfigurationError("OCI user-principal config file is unavailable")
    return resolved


def _sanitize_provider_error(
    exc: Exception,
    *,
    latency_ms: int | None = None,
) -> ProviderCallError:
    status_code = _extract_status_code(exc)
    if status_code in {401, 403}:
        category = ProviderErrorCategory.AUTH
    elif status_code == 429:
        category = ProviderErrorCategory.RATE_LIMIT
    elif status_code is None and _looks_like_auth_error(exc):
        category = ProviderErrorCategory.AUTH
    elif status_code is None:
        category = ProviderErrorCategory.TRANSPORT
    else:
        category = ProviderErrorCategory.PROVIDER
    return ProviderCallError(
        category,
        status_code=status_code,
        request_id=_extract_request_id(exc),
        latency_ms=latency_ms,
    )


def _looks_like_auth_error(exc: Exception) -> bool:
    class_name = type(exc).__name__.lower()
    module_name = type(exc).__module__.lower()
    return (
        "auth" in class_name
        or "signer" in class_name
        or module_name == "oci"
        or module_name.startswith(("oci.", "oci_genai_auth"))
    )


def _extract_status_code(exc: Exception) -> int | None:
    candidates = (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    )
    for candidate in candidates:
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and 100 <= candidate <= 599
        ):
            return candidate
    return None


def _extract_request_id(exc: Exception) -> str | None:
    candidates: list[object] = [getattr(exc, "request_id", None)]
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if isinstance(headers, Mapping):
        candidates.extend((headers.get("opc-request-id"), headers.get("x-request-id")))
    for candidate in candidates:
        if isinstance(candidate, str) and _REQUEST_ID_PATTERN.fullmatch(candidate) is not None:
            return candidate
    return None


def _response_contains_refusal(response: object) -> bool:
    output = _field(response, "output")
    if not isinstance(output, (list, tuple)):
        return False
    for item in output:
        content = _field(item, "content")
        if not isinstance(content, (list, tuple)):
            continue
        for part in content:
            if _field(part, "type") == "refusal":
                return True
    return False


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
