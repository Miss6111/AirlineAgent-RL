# AirlineAgent-RL

AirlineAgent-RL is a reinforcement learning project for long-horizon, multi-turn airline customer service agents. It builds on the Tau-Airline task setting and trains a Qwen2.5-3B-Instruct based agent to handle user lookup, flight search, booking, cancellation, reservation updates, baggage and passenger changes, and compensation certificate workflows through tool calling.

## Highlights

- Built a multi-turn tool-use training pipeline on top of veRL, SGLang, Ray, and FSDP.
- Constructed high-quality SFT data from GPT-4o / Claude Sonnet generated historical customer-service trajectories, filtering failed, inconsistent, malformed, ungrounded, and overlong traces.
- Implemented GRPO-style RL for agentic rollout with tool execution and simulator feedback.
- Added LAS, short for Length-aware Advantage Scaling, to reduce token-level gradient scale mismatch in long responses.
- Designed mutation-aware process rewards for high-risk database-writing actions such as booking, cancellation, flight updates, baggage/passenger updates, and compensation certificate issuing.
- Logged and analyzed outcome reward, process score, final reward, advantage statistics, gradient norm, forced termination, response truncation, and tool-use diagnostics.

## Method Variants

The public experiment summary uses the following names:

| Name | Meaning |
| --- | --- |
| SFT baseline | Supervised fine-tuned model evaluated without RL updates |
| plain GRPO + LAS | Binary outcome reward with Length-aware Advantage Scaling |
| normal PRM | Legacy lightweight process reward with weight 0.5 |
| mutation-aware PRM + LAS | Mutation-aware process reward rules plus LAS |

## Main 50-Step Evaluation

The following results are from API simulator evaluation with three repeated runs. They are reported as aggregate experiment evidence rather than a claim of final convergence.

| Variant | Rep | Reward | Outcome | Process | Forced Done |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT baseline | - | 0.1467 | 0.18 | - | - |
| plain GRPO + LAS | 1 | 0.1600 | 0.16 | 0.0000 | 0.58 |
| plain GRPO + LAS | 2 | 0.2000 | 0.20 | 0.0000 | 0.58 |
| plain GRPO + LAS | 3 | 0.1800 | 0.18 | 0.0000 | 0.58 |
| plain GRPO + LAS | mean | 0.1800 | 0.18 | 0.0000 | 0.58 |
| normal PRM | 1 | 0.0489 | 0.10 | -0.0523 | 0.56 |
| normal PRM | 2 | 0.1474 | 0.20 | -0.0573 | 0.56 |
| normal PRM | 3 | 0.0742 | 0.12 | -0.0416 | 0.56 |
| normal PRM | mean | 0.0901 | 0.14 | -0.0504 | 0.56 |
| mutation-aware PRM + LAS | 1 | 0.2144 | 0.28 | -0.0832 | 0.64 |
| mutation-aware PRM + LAS | 2 | 0.1800 | 0.24 | -0.0700 | 0.70 |
| mutation-aware PRM + LAS | 3 | 0.1370 | 0.20 | -0.0760 | 0.66 |
| mutation-aware PRM + LAS | mean | 0.1771 | 0.24 | -0.0764 | 0.6667 |

See [docs/experiments.md](docs/experiments.md) for the same table and interpretation notes.

## Repository Layout

```text
docs/                              Public reward and experiment notes
examples/data_preprocess/          Tau-Airline preprocessing utilities
examples/sglang_multiturn/config/  Tool and interaction configuration
examples/sglang_multiturn/         Validation audit utility
verl/interactions/                 Tau-Airline interaction and reward logic
verl/trainer/ppo/                  GRPO/LAS and trainer integration points
verl/workers/reward_manager/       Reward aggregation and logging
verl/workers/rollout/              Multi-turn SGLang rollout integration
assets/                            Aggregate public metrics
```

## Public Scope

This public version intentionally excludes shell launch scripts, checkpoints, model weights, raw training logs, W&B files, validation transcripts, private environment files, and raw generated trajectory data.

