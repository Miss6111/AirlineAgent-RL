from __future__ import annotations

from verl.utils.reward_score import tau_airline
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager


@register("tau_airline")
class TauAirlineRewardManager(NaiveRewardManager):
    """Tau-Airline reward manager."""

    def __init__(self, tokenizer, num_examine=0):
        super().__init__(tokenizer, num_examine, compute_score=None)

    def compute_score(self, data_source, solution_str, ground_truth, extra_info):
        if data_source == "tau_airline":
            return tau_airline.compute_score(
                actions=solution_str,
                ground_truth=ground_truth,
                data=extra_info.get("data"),
                raw_data=extra_info.get("raw_data"),
                method="strict",
                format_score=0.0,
                score=1.0,
            )
        return super().compute_score(data_source, solution_str, ground_truth, extra_info)
