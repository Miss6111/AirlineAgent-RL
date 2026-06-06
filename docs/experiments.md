# Experiment Summary

This document records public aggregate results for the main 50-step ablation. It avoids exposing private launch scripts, local paths, API configuration, raw transcripts, checkpoints, or training logs.

## Naming

| Public name | Internal meaning |
| --- | --- |
| SFT baseline | Supervised fine-tuned policy evaluated directly |
| plain GRPO + LAS | Binary outcome reward with Length-aware Advantage Scaling |
| normal PRM | Legacy lightweight process reward at weight 0.5 |
| mutation-aware PRM + LAS | Mutation-aware process reward plus LAS |

## API Simulator, 50-Step, Three Repeats

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

## Interpretation

The mutation-aware PRM + LAS variant improved mean outcome from 0.18 to 0.24 versus the plain GRPO + LAS baseline in this 50-step ablation. The process score is negative by design because mutation-aware PRM penalizes unsafe or ungrounded database-writing behavior while preserving the binary outcome signal.

The project also used trajectory audits to inspect forced termination, malformed tool calls, tool execution mismatch, and no-op database matches. Those diagnostics are useful for understanding evaluation quality and are not presented as headline model performance.

