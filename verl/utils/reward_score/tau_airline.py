from __future__ import annotations

import copy
import inspect
import json
from hashlib import sha256
from typing import Any, Union

from verl.interactions.tau_airline_data import load_data
from verl.tools.tau_airline._logic import ACTION_DISPATCH


ToHashable = Union[str, int, float, dict[str, "ToHashable"], list["ToHashable"], set["ToHashable"]]
Hashable = Union[str, int, float, tuple["Hashable"], tuple[tuple[str, "Hashable"]]]
RESPOND_ACTION_NAME = "respond"


def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    if isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    if isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    return item


def consistent_hash(value: Hashable) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def step(action: dict[str, Any], raw_data: dict[str, Any] | None) -> None:
    if not isinstance(raw_data, dict):
        return
    name = action["name"]
    kwargs = action.get("kwargs", {}) or {}
    func = ACTION_DISPATCH.get(name)
    if func is None:
        return
    sig = inspect.signature(func)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    func(raw_data, **accepted)


def _restore_actions(actions_raw: Any) -> list[dict[str, Any]]:
    parsed = json.loads(actions_raw) if isinstance(actions_raw, str) else actions_raw
    out = []
    for act in parsed or []:
        kwargs = act.get("kwargs", {})
        if isinstance(kwargs, str):
            try:
                kwargs = json.loads(kwargs)
            except json.JSONDecodeError:
                kwargs = {}
        out.append({**act, "kwargs": kwargs})
    return out


def compute_score(
    actions: Any = None,
    ground_truth: Any = None,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
    data: dict[str, Any] | None = None,
    raw_data: dict[str, Any] | None = None,
    **kwargs,
) -> float:
    if raw_data is None:
        raw_data = load_data()
    if data is None:
        data = raw_data

    def get_data_hash(value: dict[str, Any]) -> str:
        return consistent_hash(to_hashable(value))

    data_hash = get_data_hash(copy.deepcopy(data))
    raw_data_copy = copy.deepcopy(raw_data)

    gt_actions = [
        action for action in _restore_actions(ground_truth) if action.get("name") != RESPOND_ACTION_NAME
    ]
    for action in gt_actions:
        step(action, raw_data_copy)

    return 1.0 if data_hash == get_data_hash(raw_data_copy) else 0.0
