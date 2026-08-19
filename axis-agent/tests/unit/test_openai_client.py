from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from axis_agent.config import AxisSettings, OciAuthMode
from axis_agent.openai_client import (
    OpenAIClientConfigurationError,
    OpenAIClientRuntime,
    OpenAIClientStateError,
    OpenAISdkBindings,
    ProviderCallError,
    ProviderErrorCategory,
    StructuredOutputClient,
    StructuredOutputError,
    StructuredOutputIssue,
    load_openai_sdk_bindings,
)


class SampleOutput(BaseModel):
    value: str


class FakeResponses:
    def __init__(self, result: object) -> None:
        self.result = result
        self.error: Exception | None = None
        self.calls: list[dict[str, object]] = []

    async def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class FakeOpenAIClient:
    def __init__(self, responses: FakeResponses) -> None:
        self.responses = responses
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeHttpClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeFactories:
    def __init__(self, response: object | None = None) -> None:
        parsed_response = response or SimpleNamespace(
            output_parsed=SampleOutput(value="ok"), output=[]
        )
        self.responses = FakeResponses(parsed_response)
        self.http_client = FakeHttpClient()
        self.openai_client = FakeOpenAIClient(self.responses)
        self.loaded_modes: list[OciAuthMode] = []
        self.http_calls: list[dict[str, object]] = []
        self.openai_calls: list[dict[str, object]] = []
        self.auth_calls: list[dict[str, object]] = []
        self.raise_openai_constructor = False

    def load(self, auth_mode: OciAuthMode) -> OpenAISdkBindings:
        self.loaded_modes.append(auth_mode)
        return OpenAISdkBindings(
            create_http_client=self.create_http_client,
            create_openai_client=self.create_openai_client,
            create_http_timeout=lambda **kwargs: ("timeout", kwargs),
            create_http_limits=lambda **kwargs: ("limits", kwargs),
            create_user_principal_auth=self.create_auth,
        )

    def create_http_client(self, **kwargs: object) -> FakeHttpClient:
        self.http_calls.append(kwargs)
        return self.http_client

    def create_openai_client(self, **kwargs: object) -> FakeOpenAIClient:
        self.openai_calls.append(kwargs)
        if self.raise_openai_constructor:
            raise RuntimeError("synthetic constructor detail")
        return self.openai_client

    def create_auth(self, **kwargs: object) -> object:
        self.auth_calls.append(kwargs)
        return object()


def api_key_settings() -> AxisSettings:
    return AxisSettings(
        _env_file=None,
        model_mode="oci_openai",
        oci_project_ocid="ocid1.generativeaiproject.oc1.iad.exampleproject",
        oci_genai_api_key="synthetic-unit-test-key",
    )


def user_principal_settings(config_file: Path) -> AxisSettings:
    return AxisSettings(
        _env_file=None,
        model_mode="oci_openai",
        oci_auth_mode="user_principal",
        oci_project_ocid="ocid1.generativeaiproject.oc1.iad.exampleproject",
        oci_config_file=config_file,
        oci_profile="AXIS_TEST",
    )


def test_sdk_loader_imports_oci_auth_only_for_user_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported: list[str] = []

    def constructor(**kwargs: object) -> object:
        return kwargs

    modules = {
        "httpx": SimpleNamespace(
            AsyncClient=constructor,
            Timeout=constructor,
            Limits=constructor,
        ),
        "openai": SimpleNamespace(AsyncOpenAI=constructor),
        "oci_genai_auth": SimpleNamespace(OciUserPrincipalAuth=constructor),
    }

    def fake_import(name: str) -> object:
        imported.append(name)
        return modules[name]

    monkeypatch.setattr("axis_agent.openai_client.import_module", fake_import)

    api_key_bindings = load_openai_sdk_bindings(OciAuthMode.API_KEY)
    assert imported == ["httpx", "openai"]
    assert api_key_bindings.create_user_principal_auth is None

    imported.clear()
    user_bindings = load_openai_sdk_bindings(OciAuthMode.USER_PRINCIPAL)
    assert imported == ["httpx", "openai", "oci_genai_auth"]
    assert user_bindings.create_user_principal_auth is constructor


@pytest.mark.asyncio
async def test_api_key_runtime_uses_hardened_transport_and_direct_structured_parse() -> None:
    factories = FakeFactories()
    runtime = OpenAIClientRuntime(api_key_settings(), sdk_loader=factories.load)

    assert isinstance(runtime, StructuredOutputClient)
    async with runtime:
        output = await runtime.parse_structured(
            model="xai.grok-4.3",
            input="sanitized task",
            instructions="return the contract",
            output_type=SampleOutput,
            max_output_tokens=512,
        )

    assert output == SampleOutput(value="ok")
    assert factories.loaded_modes == [OciAuthMode.API_KEY]
    assert factories.auth_calls == []
    assert len(factories.http_calls) == 1
    http_call = factories.http_calls[0]
    assert http_call["follow_redirects"] is False
    assert http_call["trust_env"] is False
    assert "auth" not in http_call
    assert len(factories.openai_calls) == 1
    openai_call = factories.openai_calls[0]
    assert openai_call["api_key"] == "synthetic-unit-test-key"
    assert openai_call["project"] == "ocid1.generativeaiproject.oc1.iad.exampleproject"
    assert openai_call["max_retries"] == 2
    parse_call = factories.responses.calls[0]
    assert parse_call["store"] is False
    assert parse_call["text_format"] is SampleOutput
    assert parse_call["instructions"] == "return the contract"
    assert parse_call["max_output_tokens"] == 512
    assert parse_call["timeout"] == 120.0
    assert factories.openai_client.closed is True


@pytest.mark.asyncio
async def test_user_principal_mode_uses_only_signed_http_auth(tmp_path: Path) -> None:
    config_file = tmp_path / "oci-config"
    config_file.write_text("[AXIS_TEST]\n", encoding="utf-8")
    factories = FakeFactories()

    async with OpenAIClientRuntime(user_principal_settings(config_file), sdk_loader=factories.load):
        pass

    assert factories.loaded_modes == [OciAuthMode.USER_PRINCIPAL]
    assert factories.auth_calls == [
        {"config_file": str(config_file.resolve()), "profile_name": "AXIS_TEST"}
    ]
    auth = factories.http_calls[0]["auth"]
    assert auth is not None
    assert factories.openai_calls[0]["api_key"] == "not-used"


@pytest.mark.asyncio
async def test_user_principal_missing_config_fails_without_auth_fallback(tmp_path: Path) -> None:
    factories = FakeFactories()
    runtime = OpenAIClientRuntime(
        user_principal_settings(tmp_path / "missing"), sdk_loader=factories.load
    )

    with pytest.raises(OpenAIClientConfigurationError, match="config file is unavailable"):
        await runtime.__aenter__()

    assert factories.auth_calls == []
    assert factories.http_calls == []
    assert factories.openai_calls == []


@pytest.mark.asyncio
async def test_runtime_requires_open_lifecycle() -> None:
    runtime = OpenAIClientRuntime(api_key_settings(), sdk_loader=FakeFactories().load)

    with pytest.raises(OpenAIClientStateError, match="not open"):
        await runtime.parse_structured(
            model="xai.grok-4.3",
            input="task",
            output_type=SampleOutput,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "issue"),
    [
        (SimpleNamespace(output_parsed=None, output=[]), StructuredOutputIssue.EMPTY),
        (
            SimpleNamespace(
                output_parsed=None,
                output=[{"content": [{"type": "refusal", "refusal": "sensitive"}]}],
            ),
            StructuredOutputIssue.REFUSAL,
        ),
        (
            SimpleNamespace(output_parsed={"value": "not-validated"}, output=[]),
            StructuredOutputIssue.WRONG_TYPE,
        ),
    ],
)
async def test_unacceptable_structured_outputs_fail_closed(
    response: object, issue: StructuredOutputIssue
) -> None:
    factories = FakeFactories(response)

    async with OpenAIClientRuntime(api_key_settings(), sdk_loader=factories.load) as runtime:
        with pytest.raises(StructuredOutputError) as exc_info:
            await runtime.parse_structured(
                model="xai.grok-4.3",
                input="task",
                output_type=SampleOutput,
            )

    assert exc_info.value.issue is issue
    assert "sensitive" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_schema_validation_failure_is_retryable_structured_output_error() -> None:
    factories = FakeFactories()
    try:
        SampleOutput.model_validate({})
    except ValidationError as validation_error:
        factories.responses.error = validation_error

    async with OpenAIClientRuntime(api_key_settings(), sdk_loader=factories.load) as runtime:
        with pytest.raises(StructuredOutputError) as exc_info:
            await runtime.parse_structured(
                model="xai.grok-4.3",
                input="task",
                output_type=SampleOutput,
            )

    assert exc_info.value.issue is StructuredOutputIssue.SCHEMA


class SensitiveProviderError(RuntimeError):
    status_code = 429
    request_id = "req_safe-123"


@pytest.mark.asyncio
async def test_provider_errors_are_classified_and_sanitized() -> None:
    factories = FakeFactories()
    factories.responses.error = SensitiveProviderError("secret response body")

    async with OpenAIClientRuntime(api_key_settings(), sdk_loader=factories.load) as runtime:
        with pytest.raises(ProviderCallError) as exc_info:
            await runtime.parse_structured(
                model="xai.grok-4.3",
                input="task",
                output_type=SampleOutput,
            )

    error = exc_info.value
    assert error.category is ProviderErrorCategory.RATE_LIMIT
    assert error.status_code == 429
    assert error.request_id == "req_safe-123"
    assert error.latency_ms is not None
    assert error.latency_ms >= 0
    assert "secret response body" not in str(error)
    assert error.__suppress_context__ is True


@pytest.mark.asyncio
async def test_client_constructor_failure_closes_transport_and_is_sanitized() -> None:
    factories = FakeFactories()
    factories.raise_openai_constructor = True
    runtime = OpenAIClientRuntime(api_key_settings(), sdk_loader=factories.load)

    with pytest.raises(ProviderCallError) as exc_info:
        await runtime.__aenter__()

    assert exc_info.value.category is ProviderErrorCategory.TRANSPORT
    assert "synthetic constructor detail" not in str(exc_info.value)
    assert factories.http_client.closed is True
