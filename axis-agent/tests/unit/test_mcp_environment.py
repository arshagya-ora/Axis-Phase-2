from axis_agent.mcp import build_child_environment


def test_child_environment_keeps_only_required_operating_system_values() -> None:
    source = {
        "PATH": "safe-path",
        "TEMP": "safe-temp",
        "SYSTEMROOT": "safe-root",
        "AXIS_LOG_LEVEL": "DEBUG",
        "UNRELATED": "not-inherited",
    }

    assert build_child_environment(source) == {
        "PATH": "safe-path",
        "TEMP": "safe-temp",
        "SYSTEMROOT": "safe-root",
    }


def test_child_environment_omits_oci_project_model_and_key_material() -> None:
    source = {
        "PATH": "safe-path",
        "OCI_CONFIG_FILE": "C:/secret/config",
        "OCI_CLI_PROFILE": "customer",
        "OPENAI_API_KEY": "secret-key",
        "OPENAI_PROJECT": "project-id",
        "OPENAI_MODEL": "model-id",
        "CUSTOM_KEY": "secret",
        "SERVICE_TOKEN": "secret",
        "DATABASE_PASSWORD": "secret",
    }

    assert build_child_environment(source) == {"PATH": "safe-path"}


def test_child_environment_matching_is_case_insensitive() -> None:
    source = {
        "Path": "safe-path",
        "oci_config_file": "secret",
        "project_id": "secret",
        "model_name": "secret",
        "api_key": "secret",
    }

    assert build_child_environment(source) == {"Path": "safe-path"}
