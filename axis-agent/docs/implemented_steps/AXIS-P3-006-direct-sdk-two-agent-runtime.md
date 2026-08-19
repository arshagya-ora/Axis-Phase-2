# AXIS-P3-006 implemented steps

- Implementation status: complete for the planned code slice
- Deterministic regression status: verified locally (`375` offline tests; `9` deselected)
- Browser/MCP status: complete controlled-browser suite passed `9/9`
- Live OCI and complete external end-to-end status: not run
- Planned package: [P3-006 plan](../planned_steps/AXIS-P3-006-direct-sdk-two-agent-runtime.md)

| Step | Delivered behavior | Linked implementation | Verification |
|---|---|---|---|
| 1 | Minimal structured model outputs without runtime authority | [`contracts/model_io.py`](../../src/axis_agent/contracts/model_io.py) | Model-contract tests |
| 2 | Policy-bounded direct-SDK Planner with runtime-owned plan materialization | [`planner.py`](../../src/axis_agent/planner.py) | Planner tests |
| 3 | One-action direct-SDK Navigator bound to a fresh observation | [`navigator.py`](../../src/axis_agent/navigator.py) | Navigator tests |
| 4 | Hardened direct `AsyncOpenAI` lifecycle with both explicit OCI auth modes | [`openai_client.py`](../../src/axis_agent/openai_client.py), [`config.py`](../../src/axis_agent/config.py) | Client/configuration tests; live schema gate not run |
| 5 | Navigator-only task-scoped MCP with trusted step binding and ephemeral Playwright | [`mcp/direct_client.py`](../../src/axis_agent/mcp/direct_client.py), [`mcp/application.py`](../../src/axis_agent/mcp/application.py) | Unit coverage and [`test_direct_mcp_runtime.py`](../../tests/integration/test_direct_mcp_runtime.py) passed locally |
| 6 | Live application composition and explicit live/mock CLI modes | [`app.py`](../../src/axis_agent/app.py), [`cli.py`](../../src/axis_agent/cli.py), [`devtools/mock_runtime.py`](../../src/axis_agent/devtools/mock_runtime.py) | Application/CLI tests |
| 7 | Firewall-trusted current-origin auto-authorization, credential/payment/OTP/token field denial, and dispatcher/network enforcement | [`runtime/approvals.py`](../../src/axis_agent/runtime/approvals.py), [`runtime/dispatcher.py`](../../src/axis_agent/runtime/dispatcher.py), [`browser/playwright.py`](../../src/axis_agent/browser/playwright.py) | Approval, dispatcher, and complete `9/9` controlled-browser suite |
| 8 | Agents SDK retired and release documentation aligned | [`pyproject.toml`](../../pyproject.toml), [`README.md`](../../README.md), [Phase 3 acceptance](../PHASE_3_ACCEPTANCE.md) | Dependency/import, Markdown link, and secret-pattern checks |

## External evidence still required

- Run the exact-schema live OCI gate with a newly rotated API key or reviewed User Principal
  configuration.
- Run one controlled direct-SDK-to-browser synthetic task.
- Record deployment egress, secret-manager, isolation, monitoring, rollback, and security-review
  evidence before customer release.
