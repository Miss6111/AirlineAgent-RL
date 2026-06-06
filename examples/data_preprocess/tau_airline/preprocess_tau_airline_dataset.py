"""Preprocess Tau Airline tasks to verl parquet format."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from datasets import Dataset
from tqdm import tqdm

from external.tau_bench_airline.tasks import tasks as TASKS_TRAIN
from external.tau_bench_airline.tasks_test import TASKS as TASKS_TEST
from external.tau_bench_airline.wiki import WIKI
from verl.tools.tau_airline._logic import TOOL_CLASS_BY_NAME
from verl.utils.hdfs_io import copy, makedirs


def make_action_serializable(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        name = action["name"]
        kwargs = action.get("arguments", action.get("kwargs", {}))
    else:
        name = action.name
        kwargs = action.kwargs
    return {
        "name": name,
        "kwargs": json.dumps(kwargs, ensure_ascii=False),
    }


def task_field(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(key, default)
    return getattr(task, key, default)


def build_tools_kwargs(gt_actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        tool_name: {"create_kwargs": {"ground_truth": gt_actions}}
        for tool_name in TOOL_CLASS_BY_NAME
    }


def build_row(split: str, idx: int, task: Any) -> dict[str, Any]:
    gt_actions = [make_action_serializable(action) for action in task_field(task, "actions", [])]
    instruction = task_field(task, "instruction", "")
    user_id = task_field(task, "user_id", "")
    return {
        "data_source": "tau_airline",
        "agent_name": "airline_agent",
        "prompt": [
            {
                "role": "system",
                "content": WIKI,
            }
        ],
        "ability": "tau_airline",
        "reward_model": {"style": "rule", "ground_truth": gt_actions},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": "",
            "question": instruction,
            "need_tools_kwargs": True,
            "tools_kwargs": build_tools_kwargs(gt_actions),
            "interaction_kwargs": {
                "query": instruction,
                "user_id": user_id,
                "ground_truth": gt_actions,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/tau_airline")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    train_rows = [
        build_row("train", idx, task)
        for idx, task in tqdm(enumerate(TASKS_TRAIN), total=len(TASKS_TRAIN), desc="Processing train")
    ]
    test_rows = [
        build_row("test", idx, task)
        for idx, task in tqdm(enumerate(TASKS_TEST), total=len(TASKS_TEST), desc="Processing test")
    ]

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    Dataset.from_list(train_rows).to_parquet(os.path.join(local_dir, "train.parquet"))
    Dataset.from_list(test_rows).to_parquet(os.path.join(local_dir, "test.parquet"))

    print(f"train dataset len : {len(train_rows)}")
    print(f"test dataset len : {len(test_rows)}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
