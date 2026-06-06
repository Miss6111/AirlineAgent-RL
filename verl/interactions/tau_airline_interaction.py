import copy
import asyncio
import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from litellm import (
    APIConnectionError,
    APIError,
    BadRequestError,
    BadGatewayError,
    InternalServerError,
    OpenAIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    completion,
)

from verl.utils.reward_score import tau_airline

from .base import BaseInteraction
from .tau_airline_data import load_data


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


USER_SIM_RETRYABLE_ERRORS = (
    Timeout,
    APIConnectionError,
    RateLimitError,
    OpenAIError,
    APIError,
    ServiceUnavailableError,
    InternalServerError,
    BadGatewayError,
)


def _is_retryable_user_sim_error(exc: Exception) -> bool:
    if isinstance(exc, USER_SIM_RETRYABLE_ERRORS):
        return True
    if not isinstance(exc, BadRequestError):
        return False
    error_text = str(exc).lower()
    return "<html" in error_text and ("<center>alb</center>" in error_text or "400 bad request" in error_text)


_READ_TOOLS = frozenset({
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "get_user_details",
    "get_reservation_details",
    "calculate",
})
_WRITE_TOOLS = frozenset({
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "update_reservation_flights",
    "send_certificate",
})
_ESCALATION_TOOLS = frozenset({"transfer_to_human_agents"})
_THINK_TOOLS = frozenset({"think", "implicit_think"})
_PRM_RULE_VERSION_CODES = {
    "legacy": 0.0,
    "mutation_v1": 1.0,
}
_MUTATING_REQUIRED_FIELDS = {
    "book_reservation": (
        "user_id",
        "origin",
        "destination",
        "flight_type",
        "cabin",
        "flights",
        "passengers",
        "payment_methods",
        "total_baggages",
        "nonfree_baggages",
        "insurance",
    ),
    "cancel_reservation": ("reservation_id",),
    "update_reservation_baggages": ("reservation_id", "total_baggages", "nonfree_baggages", "payment_id"),
    "update_reservation_flights": ("reservation_id", "cabin", "flights", "payment_id"),
    "update_reservation_passengers": ("reservation_id", "passengers"),
    "send_certificate": ("user_id", "amount"),
}
_MUTATING_PRM_KEYS = (
    "prm/response_tool_mixing_penalty",
    "prm/ask_then_tool_penalty",
    "prm/confirm_then_write_without_wait_penalty",
    "prm/ask_then_wait_bonus",
    "prm/write_without_confirmation_penalty",
    "prm/ungrounded_mutating_args_penalty",
    "prm/fabricated_flight_penalty",
    "prm/ungrounded_payment_id_penalty",
    "prm/missing_required_mutating_arg_penalty",
    "prm/missing_payment_id_penalty",
    "prm/premature_mutating_action_penalty",
    "prm/certificate_policy_penalty",
    "prm/safe_mutating_sequence_bonus",
    "prm/malformed_tool_call_penalty",
    "prm/unexecuted_tool_call_penalty",
)

_PARAM_PATTERNS = {
    "reservation_id": re.compile(r"^[A-Z0-9]{6}$"),
    "user_id": re.compile(r"^[a-z]+_[a-z]+_[0-9]+$"),
    "payment_id": re.compile(r"^(credit_card|gift_card|certificate)_[0-9]+$"),
    "flight_number": re.compile(r"^[A-Z]{3}[0-9]{3}$"),
    "origin": re.compile(r"^[A-Z]{3}$"),
    "destination": re.compile(r"^[A-Z]{3}$"),
    "date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
}
_PLACEHOLDER_KEYWORDS = frozenset({
    "previous",
    "unknown",
    "placeholder",
    "none",
    "null",
    "n/a",
    "any",
    "some",
    "first",
    "last",
    "default",
    "example",
    "sample",
    "test",
    "dummy",
    "temp",
    "temporary",
})


def _parse_tool_params(function: dict[str, Any]) -> dict[str, Any]:
    params = function.get("kwargs", function.get("arguments", {})) or {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return {}
    return params if isinstance(params, dict) else {}


def _parse_text_tool_calls(content: str) -> tuple[list[dict[str, Any]], int]:
    """Best-effort parser for raw tool calls that stayed in assistant text.

    Only structured `tool_calls` are executed by rollout. Raw blocks are used
    for diagnostics and PRM penalties, not for DB-state scoring.
    """
    calls = []
    malformed_count = 0
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", content or "", re.DOTALL | re.IGNORECASE):
        block = match.group(1).strip()
        if not block:
            malformed_count += 1
            continue

        parsed: dict[str, Any] | None = None
        try:
            value = json.loads(block)
            if isinstance(value, dict):
                parsed = value
        except json.JSONDecodeError:
            parsed = None

        if parsed is None:
            name_match = re.search(r"\bname\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", block)
            args_match = re.search(r"\barguments\s*:\s*(\{.*\})\s*$", block, re.DOTALL)
            if name_match and args_match:
                try:
                    parsed = {"name": name_match.group(1), "arguments": json.loads(args_match.group(1))}
                except json.JSONDecodeError:
                    parsed = None
            if parsed is None:
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                if len(lines) >= 2 and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", lines[0]):
                    try:
                        parsed = {"name": lines[0], "arguments": json.loads("\n".join(lines[1:]))}
                    except json.JSONDecodeError:
                        parsed = None

        if parsed is None or not isinstance(parsed.get("name"), str):
            malformed_count += 1
            continue
        params = parsed.get("arguments", parsed.get("kwargs", {}))
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
                malformed_count += 1
        if not isinstance(params, dict):
            params = {}
            malformed_count += 1
        calls.append({"name": parsed["name"], "arguments": params, "malformed": True})
    return calls, malformed_count


def _to_score_action(function: dict[str, Any]) -> dict[str, Any]:
    params = _parse_tool_params(function)
    return {"name": function.get("name", ""), "kwargs": params}


def _param_str(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False).lower()


def _is_placeholder_param(field_name: str, value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lower = value.lower()
    if any(kw in lower for kw in _PLACEHOLDER_KEYWORDS):
        return True
    for key, pattern in _PARAM_PATTERNS.items():
        if key in field_name.lower():
            return not bool(pattern.match(value))
    return False


def _has_placeholder(params: dict[str, Any]) -> bool:
    return any(_is_placeholder_param(field_name, value) for field_name, value in params.items())


def _iter_param_items(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_param_items(child, child_prefix)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            yield from _iter_param_items(child, f"{prefix}[{idx}]")
    else:
        yield prefix, value


def _collect_seen_entities(action_history: list[dict[str, Any]]) -> set[str]:
    seen_entities = set()
    for prev in action_history:
        for ent_list in prev.get("extracted_entities", {}).values():
            seen_entities.update(ent_list)
    return seen_entities


def _param_values_for_field(params: dict[str, Any], field_name: str) -> list[str]:
    return [
        value
        for path, value in _iter_param_items(params)
        if path.endswith(field_name) and isinstance(value, str)
    ]


def _has_field(params: dict[str, Any], field_name: str) -> bool:
    return bool(_param_values_for_field(params, field_name))


def _missing_required_fields(tool: str, params: dict[str, Any]) -> list[str]:
    return [
        field
        for field in _MUTATING_REQUIRED_FIELDS.get(tool, ())
        if field not in params or params[field] in (None, "", [])
    ]


def _has_ungrounded_values(params: dict[str, Any], fields: tuple[str, ...], seen_entities: set[str]) -> bool:
    values = []
    for field in fields:
        values.extend(_param_values_for_field(params, field))
    return any(value not in seen_entities for value in values)


def _asks_for_user_id(content: str) -> bool:
    lower = content.lower()
    ask_terms = ("user id", "user_id", "customer id", "account id", "profile id")
    return "?" in content and any(term in lower for term in ask_terms)


def _asks_for_confirmation(content: str) -> bool:
    lower = content.lower()
    confirm_terms = (
        "confirm",
        "confirmation",
        "please confirm",
        "can you confirm",
        "would you like me to",
        "should i",
        "is that correct",
        "are you sure",
    )
    return "?" in content and any(term in lower for term in confirm_terms)


def _asks_for_information(content: str) -> bool:
    lower = content.lower()
    ask_terms = (
        "what is",
        "which",
        "please provide",
        "can you provide",
        "could you provide",
        "tell me",
        "need your",
        "need the",
        "user id",
        "payment",
        "reservation",
        "confirm",
    )
    return "?" in content and any(term in lower for term in ask_terms)


def _is_redundant(action_history: list[dict[str, Any]], current_tool: str, current_params: dict[str, Any], window: int = 3) -> bool:
    current_sig = (current_tool, _param_str(current_params))
    for prev in action_history[-window:]:
        prev_tool = prev.get("tool", "")
        if prev_tool in _THINK_TOOLS:
            continue
        if (prev_tool, prev.get("param_str", "")) == current_sig:
            return True
    return False


def _extract_entities_from_text(text: str) -> dict[str, list[str]]:
    return {
        "reservation_id": re.findall(r"\b[A-Z0-9]{6}\b", text or ""),
        "user_id": re.findall(r"\b[a-z]+_[a-z]+_[0-9]+\b", text or ""),
        "payment_id": re.findall(r"\b(?:credit_card|gift_card|certificate)_[0-9]+\b", text or ""),
        "flight_number": re.findall(r"\b[A-Z]{3}[0-9]{3}\b", text or ""),
    }


_PRM_NORMALIZE_MODE_CODES = {
    "binary": 0.0,
    "mean_clip": 1.0,
    "sum_clip": 2.0,
    "none": 3.0,
}


def _clip_score(score: float, clip_min: float, clip_max: float) -> float:
    return max(clip_min, min(clip_max, score))


def _compute_reasoning_quality_score(
    action_history: list[dict[str, Any]],
    normalize_mode: str = "mean_clip",
    clip_min: float = -0.5,
    clip_max: float = 0.5,
) -> float:
    return _compute_reasoning_quality_parts(
        action_history,
        normalize_mode=normalize_mode,
        clip_min=clip_min,
        clip_max=clip_max,
        rule_version="legacy",
    )["process_score"]


def _compute_reasoning_quality_parts(
    action_history: list[dict[str, Any]],
    normalize_mode: str = "mean_clip",
    clip_min: float = -0.5,
    clip_max: float = 0.5,
    rule_version: str = "legacy",
) -> dict[str, float]:
    # TRICK 3: PRM-Lite process reward.
    # A lightweight rule scorer adds dense process signal to the binary Tau
    # Airline outcome reward, so GRPO groups are less likely to collapse to all
    # 0/all 1 advantages.
    if rule_version not in _PRM_RULE_VERSION_CODES:
        raise ValueError(f"Unknown Tau Airline PRM rule version: {rule_version}")
    parts = {key: 0.0 for key in _MUTATING_PRM_KEYS}
    if not action_history:
        parts["process_score"] = 0.0
        return parts

    per_step_scores = []
    for i, action in enumerate(action_history):
        tool = action.get("tool", "")
        params = action.get("parameters", {}) or {}
        pstr = action.get("param_str", "")
        score = 0.0
        content = action.get("content", "")
        has_response_text = bool(content.strip())
        is_executed = bool(action.get("executed", False))
        is_malformed_tool_call = bool(action.get("malformed_tool_call", False))
        seen_entities = _collect_seen_entities(action_history[:i])

        if is_malformed_tool_call:
            score -= 0.25
            parts["prm/malformed_tool_call_penalty"] -= 0.25

        if tool not in _THINK_TOOLS and not is_executed:
            score -= 0.20
            parts["prm/unexecuted_tool_call_penalty"] -= 0.20

        if tool not in _THINK_TOOLS and _has_placeholder(params):
            score += -0.05 if tool in _WRITE_TOOLS else -0.03

        if tool not in _THINK_TOOLS and _is_redundant(action_history[:i], tool, params, window=3):
            score -= 0.03

        if i >= 1 and tool not in _THINK_TOOLS:
            prev = action_history[i - 1]
            if prev.get("is_error", False):
                prev_sig = (prev.get("tool", ""), prev.get("param_str", ""))
                curr_sig = (tool, pstr)
                score += -0.04 if curr_sig == prev_sig else 0.05

        if tool in _ESCALATION_TOOLS:
            has_done_read = any(prev.get("tool") in _READ_TOOLS for prev in action_history[:i])
            score += -0.10 if not has_done_read else -0.05

        if i >= 1 and tool not in _THINK_TOOLS and params:
            if any(isinstance(v, str) and v in seen_entities for v in params.values()):
                score += 0.08 if tool in _WRITE_TOOLS else 0.04

        if tool in _READ_TOOLS:
            seen_reads = {prev.get("tool") for prev in action_history[:i] if prev.get("tool") in _READ_TOOLS}
            if tool not in seen_reads:
                score += 0.01

        # B4/B5 disabled: explicit/implicit think text is not a reliable
        # accuracy signal in this simulated airline-dialog setting. Keep the
        # rule documented, but do not reward generated thinking text.

        if tool not in _THINK_TOOLS:
            if 0 < len(content) < 30:
                score -= 0.02

        # P10: response + tool in the same assistant turn. For tool-use tasks,
        # the assistant should either ask/say something or call a tool, not both.
        if tool not in _THINK_TOOLS and has_response_text:
            score -= 0.15
            if rule_version == "mutation_v1":
                parts["prm/response_tool_mixing_penalty"] -= 0.15

        # P11: asking for user_id and immediately querying a made-up user_id.
        user_ids = _param_values_for_field(params, "user_id")
        if tool == "get_user_details" and _asks_for_user_id(content):
            if any(user_id not in seen_entities for user_id in user_ids):
                score -= 0.25

        # P12: asking for confirmation and executing a write in the same turn.
        if tool in _WRITE_TOOLS and _asks_for_confirmation(content):
            score -= 0.50

        # P13: update_reservation_flights must use flight numbers grounded in
        # prior flight-search/reservation results, not invented HAT-style IDs.
        if tool == "update_reservation_flights":
            flight_numbers = _param_values_for_field(params, "flight_number")
            if any(flight_number not in seen_entities for flight_number in flight_numbers):
                score -= 0.30

        if rule_version == "mutation_v1":
            if tool not in _THINK_TOOLS and has_response_text and _asks_for_information(content):
                score -= 0.10
                parts["prm/ask_then_tool_penalty"] -= 0.10

            if tool in _WRITE_TOOLS and _asks_for_confirmation(content):
                parts["prm/confirm_then_write_without_wait_penalty"] -= 0.50

            if tool in _THINK_TOOLS and _asks_for_information(content):
                score += 0.03
                parts["prm/ask_then_wait_bonus"] += 0.03

            has_prior_read = any(prev.get("tool") in _READ_TOOLS for prev in action_history[:i])
            has_prior_reservation_read = any(prev.get("tool") == "get_reservation_details" for prev in action_history[:i])
            if tool in _WRITE_TOOLS:
                missing_fields = _missing_required_fields(tool, params)
                if missing_fields:
                    penalty = -0.08 * len(missing_fields)
                    score += penalty
                    parts["prm/missing_required_mutating_arg_penalty"] += penalty

                if tool in {"book_reservation", "update_reservation_baggages", "update_reservation_flights"}:
                    has_payment = _has_field(params, "payment_id")
                    if not has_payment:
                        score -= 0.20
                        parts["prm/missing_payment_id_penalty"] -= 0.20

                if not _asks_for_confirmation(content):
                    score -= 0.12
                    parts["prm/write_without_confirmation_penalty"] -= 0.12

                if not has_prior_read:
                    score -= 0.12
                    parts["prm/premature_mutating_action_penalty"] -= 0.12

                grounded_fields = ("reservation_id", "user_id", "payment_id", "flight_number")
                if _has_ungrounded_values(params, grounded_fields, seen_entities):
                    score -= 0.12
                    parts["prm/ungrounded_mutating_args_penalty"] -= 0.12

                if _has_ungrounded_values(params, ("payment_id",), seen_entities):
                    score -= 0.18
                    parts["prm/ungrounded_payment_id_penalty"] -= 0.18

                flight_numbers = _param_values_for_field(params, "flight_number")
                if flight_numbers and any(flight_number not in seen_entities for flight_number in flight_numbers):
                    score -= 0.18
                    parts["prm/fabricated_flight_penalty"] -= 0.18

                if tool == "send_certificate" and not has_prior_read:
                    score -= 0.20
                    parts["prm/certificate_policy_penalty"] -= 0.20

                if (
                    not missing_fields
                    and has_prior_read
                    and (tool == "book_reservation" or has_prior_reservation_read)
                    and not _has_ungrounded_values(
                        params,
                        ("reservation_id", "user_id", "payment_id", "flight_number"),
                        seen_entities,
                    )
                ):
                    score += 0.08
                    parts["prm/safe_mutating_sequence_bonus"] += 0.08

        per_step_scores.append(score)

    if normalize_mode not in {"mean_clip", "sum_clip", "none"}:
        raise ValueError(f"Unknown Tau Airline PRM normalize mode: {normalize_mode}")
    if clip_min > clip_max:
        raise ValueError(f"Tau Airline PRM clip_min must be <= clip_max, got {clip_min} > {clip_max}")

    if normalize_mode == "sum_clip":
        score = sum(per_step_scores)
    else:
        score = sum(per_step_scores) / len(per_step_scores)

    # P5 disabled: do not penalize efficient tool-use trajectories simply
    # because they contain no explicit think/implicit_think step.

    all_reads = {action.get("tool") for action in action_history if action.get("tool") in _READ_TOOLS}
    if len(all_reads) >= 3:
        score += 0.01

    if len(action_history) > 8:
        score -= 0.01 * (len(action_history) - 8)

    if normalize_mode == "none":
        parts["process_score"] = float(score)
        return parts
    parts["process_score"] = float(_clip_score(score, clip_min, clip_max))
    return parts


class TauAirlineInteraction(BaseInteraction):
    """Interaction adapter for Tau Airline."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.model = config.get("model", "gpt-4o")
        self.provider = config.get("provider", None)
        # TRICK 3 can be controlled from the launcher for ablations.
        # TAU_AIRLINE_REWARD_MODE=binary disables PRM-Lite and uses strict outcome reward only.
        self.reward_mode = os.getenv("TAU_AIRLINE_REWARD_MODE", config.get("reward_mode", "prm_lite"))
        self.prm_lite_weight = float(os.getenv("TAU_AIRLINE_PRM_LITE_WEIGHT", config.get("prm_lite_weight", 0.3)))
        self.prm_normalize_mode = os.getenv(
            "TAU_AIRLINE_PRM_NORMALIZE_MODE",
            config.get("prm_normalize_mode", "mean_clip"),
        )
        self.prm_clip_min = float(os.getenv("TAU_AIRLINE_PRM_CLIP_MIN", config.get("prm_clip_min", -0.5)))
        self.prm_clip_max = float(os.getenv("TAU_AIRLINE_PRM_CLIP_MAX", config.get("prm_clip_max", 0.5)))
        self.prm_rule_version = os.getenv(
            "TAU_AIRLINE_PRM_RULE_VERSION",
            config.get("prm_rule_version", "legacy"),
        )
        if self.prm_normalize_mode not in {"mean_clip", "sum_clip", "none"}:
            raise ValueError(f"Unknown Tau Airline PRM normalize mode: {self.prm_normalize_mode}")
        if self.prm_rule_version not in _PRM_RULE_VERSION_CODES:
            raise ValueError(f"Unknown Tau Airline PRM rule version: {self.prm_rule_version}")
        if self.prm_clip_min > self.prm_clip_max:
            raise ValueError(
                f"Tau Airline PRM clip_min must be <= clip_max, got {self.prm_clip_min} > {self.prm_clip_max}"
            )
        self.truncation_penalty = float(os.getenv("TAU_AIRLINE_TRUNCATION_PENALTY", config.get("truncation_penalty", 0.05)))
        self.forced_done_outcome_policy = os.getenv(
            "TAU_AIRLINE_FORCED_DONE_OUTCOME_POLICY",
            config.get("forced_done_outcome_policy", "allow"),
        )
        if self.forced_done_outcome_policy not in {"zero", "allow"}:
            raise ValueError(
                "TAU_AIRLINE_FORCED_DONE_OUTCOME_POLICY must be one of: zero, allow. "
                f"Got: {self.forced_done_outcome_policy}"
            )
        self.total_cost = 0.0
        self.last_tool_error = False
        self.max_user_sim_retries = int(os.getenv("TAU_AIRLINE_USER_SIM_RETRIES", "3"))

    async def start_interaction(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        restored_gt = [
            {**act, "kwargs": json.loads(act["kwargs"])}
            if isinstance(act.get("kwargs"), str) else copy.deepcopy(act)
            for act in ground_truth or []
        ]
        raw_data = load_data()
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": restored_gt,
            "reward": 0.0,
            "data": copy.deepcopy(raw_data),
            "raw_data": raw_data,
            "actions": [],
            "action_history": [],
        }
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict]:
        messages_swapped = self.swap_roles_and_replace_system(
            messages, instruction=kwargs.get("query", ""), instance_id=instance_id
        )
        timeout = float(os.getenv("TAU_AIRLINE_USER_SIM_TIMEOUT", "90"))
        task_info = self._format_task_info(instance_id, kwargs)
        res = None
        last_error: Exception | None = None

        for attempt in range(1, self.max_user_sim_retries + 1):
            try:
                res = completion(
                    model=self.model,
                    custom_llm_provider=self.provider,
                    messages=messages_swapped,
                    request_timeout=timeout,
                    timeout=timeout,
                )
                break
            except Exception as exc:
                if not _is_retryable_user_sim_error(exc):
                    raise
                last_error = exc
                logger.warning(
                    "Tau airline user simulator API error: %s attempt=%s/%s timeout=%ss error_type=%s error=%s",
                    task_info,
                    attempt,
                    self.max_user_sim_retries,
                    timeout,
                    type(exc).__name__,
                    exc,
                )
                if attempt < self.max_user_sim_retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 8))

        if res is None:
            metric_name = (
                "user_simulator_timeout"
                if isinstance(last_error, Timeout)
                else "user_simulator_api_error"
            )
            logger.error(
                "Tau airline user simulator failed after retries: %s attempts=%s metric=%s error_type=%s error=%s",
                task_info,
                self.max_user_sim_retries,
                metric_name,
                type(last_error).__name__ if last_error is not None else "unknown",
                last_error,
            )
            self._instance_dict[instance_id]["response"] = "###STOP###"
            return True, "###STOP###", 0.0, {metric_name: 1.0}

        message = res.choices[0].message
        response = message.model_dump()["content"]
        self.total_cost = res._hidden_params["response_cost"]
        self._instance_dict[instance_id]["response"] = response

        should_terminate_sequence = "###STOP###" in response
        if should_terminate_sequence:
            reward_parts = self.calculate_reward_parts(instance_id, forced_done=False)
            return should_terminate_sequence, response, reward_parts["final_reward"], reward_parts
        return should_terminate_sequence, response, 0.0, {}

    def _format_task_info(self, instance_id: str, kwargs: dict[str, Any]) -> str:
        query = kwargs.get("query") or ""
        if len(query) > 120:
            query = query[:117] + "..."
        task_id = kwargs.get("task_id") or kwargs.get("index") or kwargs.get("data_id")
        if task_id is None:
            return f"session={instance_id} query={query!r}"
        return f"session={instance_id} task={task_id} query={query!r}"

    def calculate_reward_parts(self, instance_id: str, forced_done: bool = False) -> dict[str, float]:
        db_state_match = tau_airline.compute_score(
            actions=self._instance_dict[instance_id]["actions"],
            ground_truth=self._instance_dict[instance_id]["ground_truth"],
            data=copy.deepcopy(self._instance_dict[instance_id]["data"]),
            raw_data=copy.deepcopy(self._instance_dict[instance_id]["raw_data"]),
            method="strict",
            format_score=0.0,
            score=1.0,
        )
        outcome = db_state_match
        action_history = self._instance_dict[instance_id]["action_history"]
        parsed_tool_call_count = sum(1 for action in action_history if action.get("tool") not in _THINK_TOOLS)
        executed_tool_call_count = sum(
            1 for action in action_history if action.get("tool") not in _THINK_TOOLS and action.get("executed", False)
        )
        executed_mutating_tool_call_count = sum(
            1 for action in action_history if action.get("tool") in _WRITE_TOOLS and action.get("executed", False)
        )
        malformed_tool_call_count = sum(1 for action in action_history if action.get("malformed_tool_call", False))
        if self.reward_mode == "binary":
            process_score = 0.0
            process_parts = {key: 0.0 for key in _MUTATING_PRM_KEYS}
            final_reward = outcome
        elif self.reward_mode == "prm_lite":
            process_parts = _compute_reasoning_quality_parts(
                action_history,
                normalize_mode=self.prm_normalize_mode,
                clip_min=self.prm_clip_min,
                clip_max=self.prm_clip_max,
                rule_version=self.prm_rule_version,
            )
            process_score = process_parts["process_score"]
            final_reward = outcome + self.prm_lite_weight * process_score
        else:
            raise ValueError(f"Unknown Tau Airline reward_mode: {self.reward_mode}")
        return {
            "db_state_match": float(db_state_match),
            "outcome_reward": float(outcome),
            "process_score": float(process_score),
            "final_reward": float(final_reward),
            "forced_done": float(forced_done),
            "truncation_penalty": 0.0,
            "parsed_tool_call_count": float(parsed_tool_call_count),
            "executed_tool_call_count": float(executed_tool_call_count),
            "executed_mutating_tool_call_count": float(executed_mutating_tool_call_count),
            "malformed_tool_call_count": float(malformed_tool_call_count),
            "prm_lite_weight": float(self.prm_lite_weight),
            "prm_clip_min": float(self.prm_clip_min),
            "prm_clip_max": float(self.prm_clip_max),
            "reward_mode_code": float(1.0 if self.reward_mode == "prm_lite" else 0.0),
            "prm_normalize_mode_code": float(_PRM_NORMALIZE_MODE_CODES[self.prm_normalize_mode]),
            "prm_rule_version_code": float(_PRM_RULE_VERSION_CODES[self.prm_rule_version]),
            **{key: float(process_parts.get(key, 0.0)) for key in _MUTATING_PRM_KEYS},
        }

    async def finalize_forced_reward(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        instruction: Optional[str] = None,
        reason: str = "forced_done",
        **kwargs,
    ) -> tuple[float, dict[str, float]]:
        # Rebuild actions/action_history from the final transcript before
        # scoring. This covers cases where rollout ended on length/max-turns
        # before another user-simulator interaction occurred.
        self.swap_roles_and_replace_system(messages, instruction=instruction, instance_id=instance_id)
        reward_parts = self.calculate_reward_parts(instance_id, forced_done=True)
        reward_parts["forced_done_reason"] = reason
        return reward_parts["final_reward"], reward_parts

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        reward_parts = self.calculate_reward_parts(instance_id, forced_done=False)
        outcome = reward_parts["outcome_reward"]
        if self.reward_mode == "binary":
            return outcome
        if self.reward_mode != "prm_lite":
            raise ValueError(f"Unknown Tau Airline reward_mode: {self.reward_mode}")
        return reward_parts["final_reward"]

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)

    def get_data(self, instance_id: str) -> dict[str, Any]:
        return self._instance_dict.get(instance_id, {}).get("data", {})

    def swap_roles_and_replace_system(
        self,
        messages: list[dict[str, Any]],
        instance_id: str,
        instruction: Optional[str] = None,
    ) -> list[dict[str, str]]:
        new_messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt(instruction)}
        ]
        rebuilt_actions = []
        rebuilt_history = []
        pending_tool_actions: list[dict[str, Any]] = []
        last_assistant_content = ""
        malformed_tool_call_count = 0

        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            if role == "assistant":
                role = "user"
                last_assistant_content = msg.get("content", "") or ""
                if last_assistant_content and len(last_assistant_content) > 100:
                    rebuilt_history.append({
                        "tool": "implicit_think",
                        "parameters": {},
                        "param_str": "",
                        "is_error": False,
                        "extracted_entities": {},
                        "content": last_assistant_content[:300],
                    })
                if msg.get("tool_calls"):
                    actions = [tool_call.get("function") or {} for tool_call in msg.get("tool_calls")]
                    for function in actions:
                        params = _parse_tool_params(function)
                        score_action = _to_score_action(function)
                        rebuilt_actions.append(score_action)
                        history_action = {
                            "tool": score_action["name"],
                            "parameters": params,
                            "param_str": _param_str(params),
                            "is_error": False,
                            "executed": False,
                            "malformed_tool_call": False,
                            "extracted_entities": {},
                            "content": last_assistant_content[:300],
                        }
                        rebuilt_history.append(history_action)
                        pending_tool_actions.append(history_action)
                    msg = {**msg, "tool_calls": None}
                else:
                    text_calls, malformed_count = _parse_text_tool_calls(last_assistant_content)
                    malformed_tool_call_count += malformed_count
                    for function in text_calls:
                        params = _parse_tool_params(function)
                        rebuilt_history.append({
                            "tool": function.get("name", ""),
                            "parameters": params,
                            "param_str": _param_str(params),
                            "is_error": False,
                            "executed": False,
                            "malformed_tool_call": True,
                            "extracted_entities": {},
                            "content": last_assistant_content[:300],
                        })
            elif role == "user":
                role = "assistant"
            elif role == "tool":
                content = msg.get("content", "") or ""
                self.last_tool_error = "error" in content.lower()
                if pending_tool_actions:
                    history_action = pending_tool_actions.pop(0)
                    history_action["is_error"] = self.last_tool_error
                    history_action["executed"] = True
                    history_action["extracted_entities"] = _extract_entities_from_text(content)
                continue
            new_messages.append({"role": role, "content": msg.get("content", "") or ""})

        self._instance_dict[instance_id]["actions"] = rebuilt_actions
        self._instance_dict[instance_id]["action_history"] = rebuilt_history
        self._instance_dict[instance_id]["malformed_tool_call_count"] = malformed_tool_call_count
        return new_messages

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return (
            "You are a user interacting with an airline customer-service agent."
            f"{instruction_display}"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else.\n"
            "- Do not repeat the exact instruction in the conversation. Use your own words.\n"
            "- Try to make the conversation natural, and stick to the personality in the instruction."
        )
