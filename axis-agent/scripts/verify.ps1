param(
    [switch]$LiveBrowser,
    [switch]$LiveModel
)

$ErrorActionPreference = "Stop"

$AgentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $AgentRoot ".venv\Scripts\python.exe"
$PytestRunId = [Guid]::NewGuid().ToString("N")
$SystemTemp = [System.IO.Path]::GetTempPath()
$OfflineBaseTemp = Join-Path $SystemTemp "axis-agent-pytest-offline-$PytestRunId"
$LiveBaseTemp = Join-Path $SystemTemp "axis-agent-pytest-live-$PytestRunId"

if (-not (Test-Path -LiteralPath $Python)) {
    throw @"
Missing axis-agent\.venv.
Run these commands from axis-agent\:
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -e ".[dev,agent,browser]"
"@
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "==> $Label"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Push-Location $AgentRoot
try {
    # Scan only maintained Python trees. Runtime/browser/pytest artifacts can have
    # restrictive Windows ACLs and are never part of the formatting contract.
    Invoke-Python "Ruff lint" @("-m", "ruff", "check", "src", "tests")
    Invoke-Python "Ruff format check" @(
        "-m", "ruff", "format", "--check", "src", "tests"
    )
    Invoke-Python "Strict mypy" @("-m", "mypy", "src")
    Invoke-Python "Offline tests" @(
        "-m", "pytest",
        "-m", "not playwright",
        "--basetemp=$OfflineBaseTemp",
        "-p", "no:cacheprovider"
    )
    Invoke-Python "Explicit mock two-agent smoke test" @(
        "-m", "axis_agent", "agent", "run",
        "--mode", "mock",
        "--input", "Verify AXIS standalone",
        "--json"
    )

    if ($LiveBrowser) {
        try {
            Invoke-Python "Playwright dependency check" @("-c", "import playwright")
        }
        catch {
            throw @"
The Playwright Python dependency is unavailable.
Install it with:
  .\.venv\Scripts\python.exe -m pip install -e ".[dev,browser]"
"@
        }

        Write-Host "Live Chromium requires the Playwright-managed browser binary."
        Write-Host "If it is not installed, run: .\.venv\Scripts\python.exe -m playwright install chromium"
        Invoke-Python "Controlled live Chromium integration tests" @(
            "-m", "pytest", "tests/integration", "--run-playwright",
            "--basetemp=$LiveBaseTemp",
            "-p", "no:cacheprovider"
        )
    }

    if ($LiveModel) {
        $RequiredModelEnvironment = @(
            "AXIS_MODEL_MODE",
            "AXIS_OCI_BASE_URL",
            "AXIS_OCI_PROJECT_OCID",
            "AXIS_OCI_AUTH_MODE",
            "AXIS_PLANNER_MODEL",
            "AXIS_NAVIGATOR_MODEL"
        )
        $MissingModelEnvironment = @(
            $RequiredModelEnvironment | Where-Object {
                [string]::IsNullOrWhiteSpace(
                    [Environment]::GetEnvironmentVariable($_, "Process")
                )
            }
        )
        if ($MissingModelEnvironment.Count -gt 0) {
            throw (
                "LiveModel requires fresh process-level configuration: " +
                ($MissingModelEnvironment -join ", ")
            )
        }
        if ([Environment]::GetEnvironmentVariable("AXIS_MODEL_MODE", "Process") -ne "oci_openai") {
            throw "LiveModel requires AXIS_MODEL_MODE=oci_openai."
        }

        $AuthMode = [Environment]::GetEnvironmentVariable("AXIS_OCI_AUTH_MODE", "Process")
        if ($AuthMode -eq "api_key") {
            $RequiredAuthEnvironment = @("AXIS_OCI_GENAI_API_KEY")
        }
        elseif ($AuthMode -eq "user_principal") {
            $RequiredAuthEnvironment = @("AXIS_OCI_CONFIG_FILE", "AXIS_OCI_PROFILE")
            if (
                -not [string]::IsNullOrWhiteSpace(
                    [Environment]::GetEnvironmentVariable("AXIS_OCI_GENAI_API_KEY", "Process")
                )
            ) {
                throw "LiveModel user_principal mode forbids AXIS_OCI_GENAI_API_KEY."
            }
        }
        else {
            throw "LiveModel requires AXIS_OCI_AUTH_MODE=api_key or user_principal."
        }

        $MissingAuthEnvironment = @(
            $RequiredAuthEnvironment | Where-Object {
                [string]::IsNullOrWhiteSpace(
                    [Environment]::GetEnvironmentVariable($_, "Process")
                )
            }
        )
        if ($MissingAuthEnvironment.Count -gt 0) {
            throw (
                "LiveModel requires fresh process-level authentication configuration: " +
                ($MissingAuthEnvironment -join ", ")
            )
        }
        if (
            [Environment]::GetEnvironmentVariable("AXIS_LIVE_MODEL_CONFIRM", "Process") -ne
            "ROTATED_CREDENTIAL_CONFIRMED"
        ) {
            throw @"
LiveModel is disabled until a newly rotated credential is supplied in the process environment.
Set AXIS_LIVE_MODEL_CONFIRM=ROTATED_CREDENTIAL_CONFIRMED only for the explicit probe invocation.
"@
        }

        try {
            if ($AuthMode -eq "user_principal") {
                $DependencyProbe = "import httpx, mcp, oci_genai_auth, openai; import axis_agent.openai_client"
            }
            else {
                $DependencyProbe = "import httpx, mcp, openai; import axis_agent.openai_client"
            }
            Invoke-Python "Direct SDK dependency check" @("-c", $DependencyProbe)
        }
        catch {
            throw @"
The optional model dependencies are unavailable.
Install them with:
  .\.venv\Scripts\python.exe -m pip install -e ".[dev,agent,browser]"
"@
        }

        $LiveModelProbe = @'
import asyncio

from axis_agent.config import load_settings
from axis_agent.contracts import NavigatorDecision, PlanDraft
from axis_agent.openai_client import OpenAIClientRuntime


async def main() -> None:
    settings = load_settings(_env_file=None)
    async with OpenAIClientRuntime(settings) as runtime:
        plan = await runtime.parse_structured(
            model=settings.planner_model,
            instructions=(
                "Return only a schema-valid AXIS PlanDraft. Use exactly one wait step, "
                "no required domains, and no unsupported actions."
            ),
            input=(
                "Qualification task: wait briefly. The plan must have one step with key "
                "wait_briefly, order 1, one success criterion, allowed action type wait, "
                "no dependencies, no required domains, and max_attempts 1."
            ),
            output_type=PlanDraft,
            max_output_tokens=4096,
        )
        decision = await runtime.parse_structured(
            model=settings.navigator_model,
            instructions=(
                "Return only a schema-valid AXIS NavigatorDecision. The qualification "
                "task is already complete, so return no browser action."
            ),
            input=(
                "Return status task_completed, action null, a short summary, empty "
                "evidence_ids, replan_requested false, and error_code null."
            ),
            output_type=NavigatorDecision,
            max_output_tokens=1024,
        )
    if not isinstance(plan, PlanDraft) or not isinstance(decision, NavigatorDecision):
        raise RuntimeError("OCI structured Responses probe returned the wrong contract type")
    print("OCI direct Responses structured-output probes passed for both configured roles.")


asyncio.run(main())
'@
        Invoke-Python "Explicit OCI Responses capability probe" @("-c", $LiveModelProbe)
    }

    Write-Host "AXIS Agent verification passed."
}
finally {
    Pop-Location

    # A pytest directory created by another Windows security principal can carry
    # an ACL that the current account cannot delete. Each invocation gets fresh
    # OS-temp paths; remove only those exact generated paths when possible.
    foreach ($PytestBaseTemp in @($OfflineBaseTemp, $LiveBaseTemp)) {
        if (Test-Path -LiteralPath $PytestBaseTemp) {
            Remove-Item -LiteralPath $PytestBaseTemp -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
