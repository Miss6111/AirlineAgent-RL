# Reward Design

AirlineAgent-RL combines a binary task outcome reward with lightweight process rewards for multi-turn tool-use behavior. The mutation-aware PRM focuses on database-writing actions where unsafe tool calls can directly change reservations or issue compensation.

## Outcome Reward

The outcome reward is computed from Tau-Airline task success, based on the final database state relative to task-specific expected behavior.

## Normal PRM

The normal PRM includes general tool-use and argument-quality signals:

| Rule | Score |
| --- | ---: |
| Malformed tool call | -0.25 |
| Tool call not executed by the environment | -0.20 |
| Placeholder, unknown, dummy, or obviously invalid write-tool argument | -0.05 |
| Placeholder, unknown, dummy, or obviously invalid read-tool argument | -0.03 |
| Repeated same tool with same arguments in a short window | -0.03 |
| Retrying same arguments after a tool error | -0.04 |
| Changing arguments after a tool error | +0.05 |
| Transfer to human before useful read context | -0.10 |
| Transfer to human after useful read context | -0.05 |
| Write-tool argument grounded in prior tool response | +0.08 |
| Read-tool argument grounded in prior tool response | +0.04 |
| First read-only tool use | +0.01 |
| Three or more read tools in a trajectory | +0.01 |
| Long action history beyond the compact budget | -0.01 per extra action |

## Mutation-Aware PRM

The mutation-aware PRM keeps the normal PRM and adds rules for high-risk write operations:

| Rule | Score |
| --- | ---: |
| Mixing final natural-language response with tool calls in the same assistant turn | -0.15 |
| Asking for information and immediately calling a tool using unconfirmed content | -0.10 |
| Asking for confirmation and writing to the database in the same turn | -0.50 |
| Waiting after asking the user for required information or confirmation | +0.03 |
| Executing a write operation without adequate confirmation | -0.12 |
| Ungrounded write-tool arguments | -0.12 |
| Fabricated flight identifier in a write operation | -0.18 |
| Ungrounded payment identifier | -0.18 |
| Missing required argument in a write operation | -0.08 per missing argument |
| Missing `payment_id` for booking or paid reservation updates | -0.20 |
| Premature booking, cancellation, update, or certificate issuing | -0.12 |
| Issuing a certificate without prior read context | -0.20 |
| Safe write sequence with read context, confirmation, grounded arguments, and required payment info | +0.08 |

## Logged PRM Components

Mutation-aware components are logged under `prm/*`, including:

- `prm/response_tool_mixing_penalty`
- `prm/ask_then_tool_penalty`
- `prm/confirm_then_write_without_wait_penalty`
- `prm/ask_then_wait_bonus`
- `prm/write_without_confirmation_penalty`
- `prm/ungrounded_mutating_args_penalty`
- `prm/fabricated_flight_penalty`
- `prm/ungrounded_payment_id_penalty`
- `prm/missing_required_mutating_arg_penalty`
- `prm/missing_payment_id_penalty`
- `prm/premature_mutating_action_penalty`
- `prm/certificate_policy_penalty`
- `prm/safe_mutating_sequence_bonus`

