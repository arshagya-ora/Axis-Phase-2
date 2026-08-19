# AXIS-P3-006: Simplified direct-SDK two-agent runtime

- Phase: 3
- Plan status: implemented and locally verified; external release qualification remains pending

## Goal

Replace the Agents-SDK-based agent layer and mock-only CLI delivery with two clear production
modules, a direct structured OpenAI client for OCI, and a Navigator-only local Playwright MCP
runtime. Preserve the existing firewall, dispatcher, Playwright, audit, and bounded workflow
controls.

## Planned steps

1. Define minimal `PlanDraft` and `NavigatorDecision` schemas that exclude runtime IDs, selectors,
   tool connections, and execution authority.
2. Implement `planner.py` to call structured Responses, intersect requested capabilities with
   trusted policy, and materialize a runtime-owned plan with one bounded correction attempt.
3. Implement `navigator.py` to use one fresh observation and return exactly one supported action or
   terminal decision.
4. Implement one hardened lifecycle-owned `AsyncOpenAI` client with explicit API-key and OCI User
   Principal modes, safe errors, `store=False`, and no free-form fallback.
5. Implement one task-scoped local AXIS MCP subprocess for Navigator with parent-only step binding,
   observe/execute tools, an ephemeral Playwright context, and secret-free environment.
6. Wire `AxisApplication` and `agent run --mode live`; retain a separate explicit `--mode mock`
   path with zero external calls and no fallback.
7. Auto-authorize supported side-effect actions only when the current origin is explicitly allowed
   by the firewall; preserve independent target/network enforcement and audit records.
8. Retire Agents SDK source/dependencies, update tests and documentation, and run offline,
   controlled-browser, live-schema, and controlled end-to-end acceptance gates.

## Locked defaults

- Planner model: `xai.grok-4.3`
- Navigator model: `xai.grok-4.20-0309-non-reasoning`
- Browser lifecycle: one ephemeral Chromium context per task
- Browser gateway: local AXIS-owned Python MCP, used only by Navigator
- Authentication: explicit `api_key` or `user_principal`; no automatic fallback
- Credential rule: revoke any previously exposed API key before testing

## Acceptance

- Offline and controlled-browser gates pass on current code.
- Exact Planner and Navigator schemas pass the opt-in live OCI probe with reviewed credentials.
- A controlled synthetic task proves the complete live chain, cleanup, firewall denial, audit, and
  absence of mock fallback.
- No document or log contains a credential or claims external evidence that was not run.
