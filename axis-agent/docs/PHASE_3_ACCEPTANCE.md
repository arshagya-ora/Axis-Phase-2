# AXIS Agent Phase 3 Acceptance

- Delivery status: **simplified two-agent implementation delivered**
- Verification status: **offline, controlled-browser, and real MCP-process gates passed locally**
- Live OCI status: **not run**
- Production acceptance: **not yet granted**

Phase 3 now provides a direct-SDK Planner-to-Navigator workflow, a task-scoped local AXIS
Playwright MCP gateway, default-deny URL controls, trusted-site auto-execution policy, and
metadata-only audit persistence. This record separates implementation from release evidence.

This acceptance record covers only the current direct-SDK two-agent implementation.

## Delivered boundary

- Exactly two production agent modules: `planner.py` and `navigator.py`. A task, workflow, MCP
  gateway, approval policy, and dispatcher are runtime components, not additional agents.
- `PlanDraft` and `NavigatorDecision` are strict model-facing contracts without runtime IDs,
  timestamps, selectors, tool connection data, or execution authority.
- Planner has no MCP or browser access. It produces an ordered plan and intersects requested
  domains and action types with trusted runtime policy.
- Navigator consumes the active step and one sanitized observation, then returns one supported
  action or one terminal decision. One corrective structured-output retry is bounded and fail
  closed.
- The production model path uses a hardened direct `AsyncOpenAI` client and
  `responses.parse(..., text_format=<Pydantic model>)` with `store=False`. The OpenAI Agents SDK is
  not a runtime or package dependency.
- API-key and OCI User Principal modes are selected explicitly. User Principal signing is loaded
  only for that mode; authentication never silently falls back.
- Default models are `xai.grok-4.3` for Planner and
  `xai.grok-4.20-0309-non-reasoning` for Navigator. Both are configurable and must pass the live
  schema probe before release.
- `agent run --mode live` composes direct SDK Planner, direct SDK Navigator, the bounded workflow,
  SQLite audit, and one task-scoped local MCP/Playwright subprocess. `--mode mock` is explicit,
  deterministic, and reports zero external calls.
- Navigator is the only MCP client. The child exposes exactly `axis_bind_step`,
  `browser_observe`, and `browser_execute`; the model is never given the raw MCP connection.
- The MCP child owns one ephemeral Playwright context for one task and receives a secret-free,
  allowlisted environment. Shutdown runs in `finally` on success, failure, timeout, or cancellation.
- Runtime binding limits every step to its approved action types. Each observation is a one-action
  lease, and page/origin/observation freshness is rechecked before dispatch.
- URL navigation is default deny and uses a single-use permit. Playwright intercepts redirects,
  popups, frames, fetch/XHR, subresources, downloads, and WebSockets.
- Side-effect actions that require approval are automatically authorized only when their trusted
  current origin matches a firewall allow rule. Unsupported capabilities remain prohibited even
  on allowed sites.
- The Playwright adapter deterministically rejects model-authored text entry into password,
  payment-card, OTP/passcode, API-key, and authentication/access-token fields based on the target's
  field type, autocomplete, name, and ID metadata.
- SQLite migration history is unchanged and continues recording bounded, sanitized plan,
  model-call, action, firewall, and approval metadata.

## Required release evidence

The implementation is not customer-ready until all applicable gates below pass and their
metadata-only results are recorded.

### Offline gate

Run from `axis-agent/`:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
```

The gate must pass without a browser, provider request, credential, project identifier, or MCP
child. It covers Ruff, formatting, strict mypy, deterministic tests, an explicit mock workflow,
and mock-browser smoke.

### Controlled-browser gate

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -LiveBrowser
```

The suite must launch only Playwright-managed Chromium against its ephemeral test server. Denied
redirect, popup, iframe, API, subresource, WebSocket, and download destinations must receive no
uncontrolled request. Browser state must be removed at task shutdown.

### Explicit live-model gate

This gate is excluded from automatic pull-request and push CI. It requires a newly rotated
credential or a reviewed User Principal configuration, complete runtime settings, and:

```text
AXIS_LIVE_MODEL_CONFIRM=ROTATED_CREDENTIAL_CONFIRMED
```

API-key mode requires `AXIS_OCI_GENAI_API_KEY`. User Principal mode requires
`AXIS_OCI_CONFIG_FILE` and `AXIS_OCI_PROFILE` and forbids `AXIS_OCI_GENAI_API_KEY`.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1 -LiveModel
```

The gate must validate the exact `PlanDraft` and `NavigatorDecision` schemas through the direct
Responses client. It must not print or persist the credential, project identifier, response
content, provider body, or full request. Passing proves only endpoint/model/schema compatibility.

### Controlled end-to-end live gate

Using a synthetic task and a dedicated non-production allowlisted site, demonstrate:

```text
direct SDK Planner
  -> direct SDK Navigator
  -> task-scoped local MCP
  -> dispatcher and trusted-site policy
  -> firewall and Playwright
  -> metadata-only audit
```

This run must prove one-action observation leases, trusted per-step binding, denied-destination
enforcement, subprocess cleanup, evidence-backed completion, and no implicit mock fallback.

## Security acceptance matrix

| Control | Implementation evidence | Release disposition |
|---|---|---|
| Exactly two production agent modules | `planner.py`, `navigator.py`, application and import tests | Locally verified |
| No OpenAI Agents SDK runtime/dependency | Direct client and dependency/import tests | Locally verified |
| Strict ordered plan and one-action decision schemas | Model contract and Planner/Navigator tests | Locally verified |
| Planner cannot use MCP or Playwright | Module composition and application tests | Locally verified |
| Navigator alone owns one task-scoped MCP browser | Direct MCP client plus real-process Chromium integration | Locally verified |
| MCP child excludes OCI credentials and model configuration | Environment tests | Locally verified |
| Fresh observation/page/origin and active-step action bounds | Navigator, MCP, workflow, and dispatcher tests | Locally verified |
| URL firewall and browser-network interception | Firewall and controlled-browser tests | Locally verified |
| Trusted-site auto-execution | Approval and MCP application tests | Locally verified |
| Credential/payment/OTP/token field entry blocked in browser adapter | Controlled Playwright action test | Locally verified |
| Direct Responses structured outputs for both default models | Opt-in live gate | **Not run** |
| Complete direct-SDK-to-browser synthetic run | Dedicated acceptance environment | **Not run** |
| OS/container egress, secret manager, tenant isolation, monitoring | Deployment evidence | **Release-blocking outside this code slice** |

## Production acceptance decision

Phase 3 may be marked accepted only when:

1. Offline and controlled-browser gates pass on a clean machine and in hosted CI.
2. The direct-SDK structured-output probe passes for both approved model IDs using rotated or
   reviewed credentials.
3. A controlled end-to-end live task proves Planner, Navigator, local MCP, Playwright, firewall,
   audit, and cleanup behavior without mock fallback.
4. Security review accepts the trusted-site auto-execution risk and prohibits broad production
   allow-all policies.
5. Deployment owners accept egress isolation, secret management, packaging, monitoring,
   rollback, incident response, and penetration-testing controls.

Until those conditions are met, this repository is an implemented Phase 3 candidate, not a
customer-ready autonomous browser agent.

## Current evidence statement

On 2026-08-17, the current direct-SDK implementation passed Ruff lint/format, strict mypy, and the
offline gate with `375` tests passed and `9` Playwright-marked tests deselected. The complete
controlled-browser gate passed `9/9` Chromium tests. That suite includes the real stdio AXIS MCP
child, dispatcher, SQLite, and Chromium integration plus deterministic denial of text entry into a
password field.

No live OCI request, default-model structured-output qualification, or complete external
direct-SDK-to-browser task is claimed by this document. Hosted-CI and deployment evidence also
remain separate release requirements.
