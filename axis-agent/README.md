# AXIS Agent

`axis-agent` is the standalone Python control plane for secure browser automation. Its production
runtime has exactly two model roles:

- `planner.py` converts an untrusted task into a bounded, typed execution plan.
- `navigator.py` chooses one action at a time and is the only agent allowed to use the local
  AXIS-owned Playwright MCP gateway.

The live path calls OCI Generative AI through the direct OpenAI Python SDK. It does not use the
OpenAI Agents SDK, model handoffs, or an unrestricted Playwright MCP server.

## Delivery status

The simplified two-agent implementation is present, including the live and explicit mock CLI
paths. Ruff, formatting, strict mypy, `375` offline tests (`9` browser tests deselected), and the
complete `9/9` controlled Chromium suite have passed locally. The browser suite includes the real
stdio MCP child, dispatcher, SQLite, and Chromium integration. A live OCI request has **not** been
run or accepted; see [Phase 3 acceptance](docs/PHASE_3_ACCEPTANCE.md).

## Architecture

```text
TaskRequest
    |
    v
planner.py ---------------- direct AsyncOpenAI Responses structured output
    |                        (PlanDraft; no browser or MCP access)
    v
runtime-owned ExecutionPlan
    |
    v
navigator.py -------------- direct AsyncOpenAI Responses structured output
    |                        (one NavigatorDecision per fresh observation)
    v
task-scoped local AXIS MCP - axis_bind_step, browser_observe, browser_execute
    |
    v
ActionDispatcher ---------- state checks, audit, approval policy, firewall permit
    |
    v
DirectPlaywrightAdapter --- ephemeral Chromium context and request interception
```

Python owns all IDs, plan-step transitions, bounds, completion decisions, approval records, and
audit state. Model outputs are proposals validated against strict Pydantic contracts.

## Security boundary

- Planner has no browser, MCP, filesystem, shell, or network tools. Requested domains and action
  types are intersected with trusted runtime policy.
- Navigator receives sanitized observations and can return only one supported action or one
  terminal decision. Selectors, XPath, JavaScript, shell, upload, download, clipboard, screenshot,
  credential-entry tools, and arbitrary networking are not exposed.
- Navigator starts one local MCP subprocess lazily for a task. The subprocess owns one ephemeral
  Playwright context, accepts trusted per-step bindings, and is closed in every exit path.
- The MCP child receives an allowlisted environment that excludes OCI credentials, project data,
  model IDs, tokens, and unrelated process variables.
- Every URL target is checked by the default-deny firewall. Playwright independently intercepts
  redirects, popups, frames, fetch/XHR, subresources, downloads, and WebSockets.
- Supported side-effect actions are automatically approved only when the trusted current origin
  matches a configured firewall allow rule. Wildcard allow-all policies are forbidden for
  production. A listed origin is therefore a trusted execution zone and may permit consequential
  clicks or form submissions.
- Immediately before entering model-authored text, the Playwright adapter rejects password,
  payment-card, OTP/passcode, API-key, and authentication/access-token fields using trusted field
  type, autocomplete, name, and ID metadata. Firewall trust cannot override this prohibition.
- One observation authorizes at most one action. Stale page, origin, observation, plan, step,
  ordinal, and replay checks fail before browser execution.
- SQLite stores bounded metadata, not raw prompts, plan prose, DOM snapshots, screenshots, form
  values, model reasoning, URL queries, credentials, or provider response bodies.

Application-level interception is defense in depth. Customer deployment still requires OS or
container egress controls, tenant/process isolation, a production secret manager, resource
limits, signed artifacts, dependency/SBOM review, monitoring, and penetration testing.

## Requirements and installation

- CPython 3.12
- Playwright-managed Chromium for live browser execution
- OCI project access through either a rotated API key or OCI User Principal authentication

From `axis-agent/` in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,agent,browser]"
.\.venv\Scripts\python.exe -m playwright install chromium
```

Dependencies and browser binaries are not vendored. Use approved package and browser mirrors in
restricted environments.

## Live configuration

Common live settings:

```text
AXIS_MODEL_MODE=oci_openai
AXIS_OCI_BASE_URL=https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com/openai/v1
AXIS_OCI_PROJECT_OCID=<runtime project OCID>
AXIS_OCI_AUTH_MODE=api_key|user_principal
AXIS_PLANNER_MODEL=xai.grok-4.3
AXIS_NAVIGATOR_MODEL=xai.grok-4.20-0309-non-reasoning
AXIS_FIREWALL_POLICY_PATH=<reviewed policy file>
```

The listed models are defaults and can be overridden explicitly. There is no automatic model or
authentication fallback.

API-key mode additionally requires:

```text
AXIS_OCI_GENAI_API_KEY=<newly rotated runtime secret>
```

User Principal mode uses:

```text
AXIS_OCI_CONFIG_FILE=<default: ~/.oci/config>
AXIS_OCI_PROFILE=<default: DEFAULT>
```

The pinned distribution is `oci-genai-auth`; its Python import is `oci_genai_auth`. API-key mode
does not import this optional signing module.

Do not set `AXIS_OCI_GENAI_API_KEY` in User Principal mode. Any credential previously pasted into
chat, source, logs, command history, or test output must be revoked and replaced before testing.
Never put a real credential or project OCID in `.env.example`.

## Run the two-agent workflow

Live execution is explicit and has no fallback:

```powershell
.\.venv\Scripts\axis-agent.exe agent run `
  --mode live `
  --input "Perform the approved browser task" `
  --start-url "https://allowed.example/path" `
  --json
```

The command requires `AXIS_MODEL_MODE=oci_openai`, a reviewed firewall policy with at least one
allow rule, valid authentication, and the Playwright dependencies.

Offline mock execution is also explicit:

```powershell
.\.venv\Scripts\axis-agent.exe agent run `
  --mode mock `
  --input "Verify AXIS standalone" `
  --json
```

Mock mode is implemented under `devtools/`, reports `externalCalls: 0`, and never becomes an
implicit fallback for a failed live run. Omit `--input` to read the task from standard input and
keep it out of shell history.

## Verification

Run the required offline gate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

Run the opt-in controlled Chromium gate:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -LiveBrowser
```

The browser suite communicates only with its test-owned local server. Production firewall code
continues to reject loopback destinations.

The live model probe is an explicit release gate, not part of normal CI. Configure either auth
mode with a newly rotated credential, set:

```text
AXIS_LIVE_MODEL_CONFIRM=ROTATED_CREDENTIAL_CONFIRMED
```

and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -LiveModel
```

The probe validates the exact Planner and Navigator structured-output contracts with `store=False`.
It must not print or persist credentials, project identifiers, response content, or provider
response bodies. Passing it proves model/endpoint compatibility, not full production acceptance.

## Package map

```text
src/axis_agent/
  planner.py              direct structured Planner
  navigator.py            direct structured Navigator and MCP-only browser surface
  app.py                  live lifecycle and workflow composition
  openai_client.py        hardened AsyncOpenAI client and both OCI auth modes
  devtools/               explicit deterministic mock runtime
  contracts/              model-facing and runtime-owned schemas
  mcp/                    task-scoped local AXIS Playwright gateway
  browser/                sanitized DOM and direct Playwright adapter
  firewall/               URL/DNS policy and navigation permits
  runtime/                workflow, approval, and dispatch controls
  persistence/            metadata-only SQLite audit store
  cli.py                  standalone live/mock entry point
```

Review evidence:

- [Phase 3 acceptance](docs/PHASE_3_ACCEPTANCE.md)
- [Planned steps](docs/planned_steps/README.md)
- [Implemented steps](docs/implemented_steps/README.md)

The direct Python SDK call uses the Responses structured-output interface documented in the
[official OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs).
