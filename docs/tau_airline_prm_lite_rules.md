# Tau Airline PRM-Lite 规则说明

本文档记录 Tau Airline GRPO 中 Trick 3 PRM-Lite 使用的规则型过程奖励。

实现位置：

- `verl/interactions/tau_airline_interaction.py`
- 函数：`_compute_reasoning_quality_score(action_history)`

最终训练奖励：

```text
binary mode:  final_reward = outcome_reward
prm_lite:     final_reward = outcome_reward + prm_lite_weight * process_score
forced done:  final_reward -= truncation_penalty when outcome_reward < 1.0
```

默认配置在：

- `examples/sglang_multiturn/config/interaction_config/tau_airline_interaction_config.yaml`

默认值保持和基线兼容：

```text
reward_mode = prm_lite
prm_lite_weight = 0.3
prm_normalize_mode = mean_clip
prm_clip_min = -0.5
prm_clip_max = 0.5
truncation_penalty = 0.05
```

这些值也可以通过环境变量覆盖：

```text
TAU_AIRLINE_REWARD_MODE
TAU_AIRLINE_PRM_LITE_WEIGHT
TAU_AIRLINE_PRM_NORMALIZE_MODE
TAU_AIRLINE_PRM_CLIP_MIN
TAU_AIRLINE_PRM_CLIP_MAX
TAU_AIRLINE_TRUNCATION_PENALTY
```


## Process Score Normalization

`_compute_reasoning_quality_score(...)` 支持 3 种 normalization mode：

| Mode | 行为 | 用途 |
| --- | --- | --- |
| `mean_clip` | 对 per-step 规则分取平均，加全局 B7/P8 修正后 clip 到 `[prm_clip_min, prm_clip_max]` | 默认行为，保持和 baseline 等价。 |
| `sum_clip` | 对 per-step 规则分求和，加全局 B7/P8 修正后 clip | 用于测试更强过程信号。 |
| `none` | 对 per-step 规则分取平均，加全局 B7/P8 修正，不做 clip | 仅建议 debug 使用。 |

`reward_parts` 中会返回这些数值字段，供 train/test 指标聚合：

```text
outcome_reward
process_score
final_reward
forced_done
truncation_penalty
prm_lite_weight
prm_clip_min
prm_clip_max
reward_mode_code
prm_normalize_mode_code
```

其中 `reward_mode_code` 使用 `binary=0`, `prm_lite=1`；`prm_normalize_mode_code` 使用 `mean_clip=1`, `sum_clip=2`, `none=3`。

## 规则表

| ID | 状态 | 分数 | 规则 | 目标作用 |
| --- | --- | ---: | --- | --- |
| P1 | 启用 | write `-0.05`, read `-0.03` | 占位符参数或看起来非法的参数 | 惩罚伪造 ID、`unknown`、`null`、示例值或 schema 不合法的参数。 |
| P2 | 启用 | `-0.03` | 最近 3 个非 think 动作内重复调用相同 tool 和相同参数 | 抑制冗余 tool spam。 |
| P3 | 启用 | `-0.04` 或 `+0.05` | tool 报错后，重复同样调用会扣分；换策略会加分 | 鼓励从错误中恢复，而不是重复失败调用。 |
| P4 | 启用 | `-0.10` 或 `-0.05` | 未充分查信息就转人工会被惩罚，完全没查信息惩罚更重 | 避免过早 escalation。 |
| B1 | 启用 | write `+0.08`, read `+0.04` | 当前参数复用了前面 tool response 中出现过的实体 | 奖励 grounded data chain。 |
| B2 | 启用 | `+0.01` | 每种 read tool 首次使用 | 鼓励有用的信息收集。 |
| B4/B5 | 禁用 | `0.00` | 旧的 think / implicit_think 奖励 | 已禁用，因为模拟对话中的 thinking 文本不是可靠准确性信号。 |
| P9 | 启用 | `-0.02` | 非 think tool call 前面的 assistant 文本过短 | 惩罚几乎没有上下文的廉价 tool 调用。 |
| P5 | 禁用 | `0.00` | 旧的无 think 轨迹惩罚 | 已禁用，避免惩罚高效且正确的 tool-use 轨迹。 |
| B7 | 启用 | `+0.01` | 轨迹中至少使用 3 种不同 read tool | 对较全面的信息收集给很小奖励。 |
| P8 | 启用 | 第 8 个动作后每个动作 `-0.01` | 轨迹过长 | 惩罚长时间试错行为。 |
| P10 | 启用 | `-0.15` | 同一 assistant turn 内同时有 response 文本和 tool call | 抑制在同一轮混合面向用户回复和 tool 执行。 |
| P11 | 启用 | `-0.25` | 一边询问 `user_id`，一边立刻用未 grounded 的 ID 调 `get_user_details` | 惩罚问用户要 ID 后又立即编造身份查询。 |
| P12 | 启用 | `-0.50` | 一边询问确认，一边同轮执行 write tool | 强惩罚未等待用户确认就执行不可逆写操作。 |
| P13 | 启用 | `-0.30` | `update_reservation_flights` 使用未 grounded 的 `flight_number` | 惩罚在航班更新写操作中编造航班号。 |

## 规则细节

### P1：占位符或非法参数惩罚

适用于非 think tool。

该规则会用占位词和轻量 schema pattern 检查参数值。例子包括：

- `unknown`
- `null`
- `placeholder`
- `example`
- 非法 `reservation_id`
- 非法 `user_id`
- 非法 `payment_id`
- 非法机场、日期、航班格式

惩罚：

- Write tools：`-0.05`
- Read tools：`-0.03`

理由：

在 airline workflow 中，错误 tool 参数和任务失败高度相关。

### P2：重复 tool 调用惩罚

适用于非 think tool。

如果最近 3 个非 think 动作中出现相同 tool 且相同参数，则扣 `-0.03`。

理由：

完全重复的调用通常表示循环或低质量探索。

### P3：错误重复与恢复

适用于前一个 tool action 的 `is_error=True` 的情况。

打分：

- 和失败动作相同 tool、相同参数：`-0.04`
- 换 tool 或换参数：`+0.05`

理由：

tool 报错后，agent 应该改变策略，而不是重复完全相同的失败调用。

### P4：转人工惩罚

适用于 `transfer_to_human_agents`。

打分：

- 之前没有任何 read tool：`-0.10`
- 之前至少有一个 read tool：`-0.05`

理由：

对于可解决任务，不应该一开始就转人工。

### B1：数据链奖励

当当前 tool 参数使用了前面 tool response 中抽取出的实体时触发。

当前抽取的实体：

- `reservation_id`
- `user_id`
- `payment_id`
- `flight_number`

打分：

- Write tools：`+0.08`
- Read tools：`+0.04`

理由：

正确完成 airline 任务通常需要复用前面工具返回的 ID 和事实。

### B2：首次 read 探索奖励

适用于 read tools。

轨迹中每种 read tool 第一次出现时，加 `+0.01`。

理由：

这是对有用信息收集的小奖励。分数保持很小，避免奖励不必要探索。

### B4/B5：Think 奖励

状态：禁用。

旧行为：

- 当 `think` 或 `implicit_think` 不是连续出现、不是最后一步，并且后面跟着非 placeholder / 非 redundant 动作时加分。
- 旧分数：`+0.01`。

禁用原因：

在这个模拟 airline 对话环境中，显式 thinking 文本不是可靠的正确性信号。奖励 thinking 可能鼓励冗长 filler，并提高截断风险，但不一定提升 tool 准确性。

### P9：廉价推理惩罚

适用于非 think tool。

如果前面的 assistant content 长度在 1 到 29 个字符之间，扣 `-0.02`。

理由：

抑制极短文本后直接 tool call 的行为，这类行为看起来像盲目试错。

### P5：无 think 轨迹惩罚

状态：禁用。

旧行为：

- 如果轨迹至少有 3 个动作，且没有 `think` 或 `implicit_think`，扣 `-0.05`。

禁用原因：

高效的 tool-use 轨迹可以没有显式 thinking 但仍然正确。惩罚无 think 轨迹可能伤害准确且简洁的行为。

### B7：Read Tool 多样性奖励

如果轨迹中至少使用 3 种不同 read tool，加 `+0.01`。

理由：

部分 airline 任务在写操作前需要多类查询。该奖励刻意保持很小。

### P8：长度惩罚

如果轨迹动作数超过 8，则应用：

```text
-0.01 * (num_actions - 8)
```

理由：

抑制过长的试错轨迹。

### P10：同轮 response + tool 惩罚

适用于非 think tool action，并且同一个 assistant message 中同时包含非空用户可见文本和 tool call。

惩罚：`-0.15`

理由：

在这个环境里，一个 turn 应该要么和用户沟通，要么执行 tool，不应该两者混在一起。混合行为经常导致不一致，尤其是文本里还在向用户要信息，但 tool call 已经继续执行。

### P11：问 user_id 后编造查询惩罚

当 assistant 文本询问 user/customer/account/profile ID，同时同轮调用 `get_user_details`，并且传入的 `user_id` 没有出现在之前 tool response 中时触发。

惩罚：`-0.25`

理由：

如果模型自己说缺 user ID，却又立刻查询一个编造 ID，这比普通非法参数更严重，因为它会制造假的身份链。

### P12：问确认后同轮写操作惩罚

当 assistant 文本询问确认，同时同轮执行 write tool 时触发。Write tools 包括：

- `book_reservation`
- `cancel_reservation`
- `update_reservation_baggages`
- `update_reservation_passengers`
- `update_reservation_flights`
- `send_certificate`

惩罚：`-0.50`

理由：

询问确认意味着模型必须等待用户回答。同一轮直接执行写操作是严重 workflow violation。

### P13：编造航班号更新惩罚

当 `update_reservation_flights` 中包含的 `flight_number` 没有出现在之前 tool response 中时触发。

惩罚：`-0.30`

理由：

航班更新应该 grounded 在之前的 `search_direct_flight`、`search_onestop_flight` 或 reservation detail 结果上。编造一个看似合法的航班号，即使 schema 合法，也可能让最终写操作错误。

## 修改规则建议

修改规则时建议遵循：

1. 优先一次只改一个规则组。
2. 保持 process-score 幅度较小，先通过 `TAU_AIRLINE_PRM_LITE_WEIGHT` 做权重 ablation。
3. reward-mode ablation 建议固定 advantage-scaling 策略，例如 LAS，不要和 advantage-estimator ablation 混在同一轮比较。
4. 避免奖励容易被 hack 的文本特征，除非它和 tool 正确性有明确相关性。
5. 对 airline 准确性来说，优先关注 grounded tool chain 和合法参数，而不是显式 reasoning 文本。
6. 每次改规则后，跑一个短 smoke，并比较：

```text
train/outcome_reward/mean
train/process_score/mean
train/process_score/max
train/process_score/min
train/final_reward/mean
train/critic/rewards/mean
train/critic/rewards/min
test/outcome_reward/mean
test/process_score/mean
test/final_reward/mean
test/forced_done/mean
```

## `mutation_v1`：Mutating-Action PRM 规则

`mutation_v1` 用于进一步约束会修改航空业务状态的工具调用。它不是一套替换版规则，而是在上述 `legacy` PRM 规则基础上继续叠加 mutating-action 规则。

### 启用方式

```bash
TAU_AIRLINE_REWARD_MODE=prm_lite
TAU_AIRLINE_PRM_LITE_WEIGHT=0.5
TAU_AIRLINE_PRM_NORMALIZE_MODE=mean_clip
TAU_AIRLINE_PRM_RULE_VERSION=mutation_v1
```

保持旧规则时使用：

```bash
TAU_AIRLINE_PRM_RULE_VERSION=legacy
```

日志中的版本编码为：

| 规则版本 | `prm_rule_version_code/mean` |
|---|---:|
| `legacy` | `0.0` |
| `mutation_v1` | `1.0` |

### 适用的 Mutating Tools

以下工具被视为 write tool。它们会创建、取消或修改业务状态，因此会进入 `mutation_v1` 的 mutating-action 检查：

| Tool |
|---|
| `book_reservation` |
| `cancel_reservation` |
| `update_reservation_baggages` |
| `update_reservation_passengers` |
| `update_reservation_flights` |
| `send_certificate` |

### 新增分项表

所有分项都会进入 `reward_parts`，并以 `prm/*` 指标聚合到训练日志和 W&B。除特别注明外，同一个 write action 可以同时触发多个分项。

| 指标 | 分值 | 触发条件 | 对 `process_score` 的作用 |
|---|---:|---|---|
| `prm/response_tool_mixing_penalty` | `-0.15` | 当前 action 不是 `think` / `implicit_think`，但同时带有可见回复文本 | 记录 `legacy` P10 的扣分，不重复扣分 |
| `prm/ask_then_tool_penalty` | `-0.10` | 当前 action 不是 think tool，同时回复文本中又向用户提问 | 新增扣分 |
| `prm/confirm_then_write_without_wait_penalty` | `-0.50` | 当前 action 调用 write tool，同时回复文本中还在向用户请求确认 | 记录 `legacy` P12 的扣分，不重复扣分 |
| `prm/ask_then_wait_bonus` | `+0.03` | 当前 action 是 think tool，且回复文本中向用户提问信息 | 新增加分 |
| `prm/write_without_confirmation_penalty` | `-0.12` | 调用 write tool，但回复文本中没有请求确认 | 新增扣分 |
| `prm/ungrounded_mutating_args_penalty` | `-0.12` | write tool 参数中的 `reservation_id`、`user_id`、`payment_id` 或 `flight_number` 存在未 grounding 的值 | 新增扣分 |
| `prm/fabricated_flight_penalty` | `-0.18` | write tool 参数含有未 grounding 的 `flight_number` | 新增扣分 |
| `prm/ungrounded_payment_id_penalty` | `-0.18` | write tool 参数含有未 grounding 的 `payment_id` | 新增扣分 |
| `prm/missing_required_mutating_arg_penalty` | `-0.08 * 缺失字段数` | write tool 缺少其必需参数 | 新增扣分 |
| `prm/missing_payment_id_penalty` | `-0.20` | `book_reservation`、`update_reservation_baggages` 或 `update_reservation_flights` 中没有 `payment_id` | 新增扣分 |
| `prm/premature_mutating_action_penalty` | `-0.12` | 调用 write tool 前没有任何 read tool 调用 | 新增扣分 |
| `prm/certificate_policy_penalty` | `-0.20` | 调用 `send_certificate` 前没有任何 read tool 调用 | 新增扣分 |
| `prm/safe_mutating_sequence_bonus` | `+0.08` | write tool 满足下文的安全调用序列条件 | 新增加分 |

其中两个日志项需要特别区分：

- `prm/response_tool_mixing_penalty` 是 `legacy` P10 的指标拆解。P10 已经对轨迹分数扣除 `-0.15`，`mutation_v1` 只额外记录分项，不会再扣一次。
- `prm/confirm_then_write_without_wait_penalty` 是 `legacy` P12 的指标拆解。P12 已经对轨迹分数扣除 `-0.50`，`mutation_v1` 只额外记录分项，不会再扣一次。

### 必需参数表

`prm/missing_required_mutating_arg_penalty` 按以下字段检查。每缺少一个字段扣 `-0.08`：

| Tool | 必需参数 |
|---|---|
| `book_reservation` | `user_id`、`origin`、`destination`、`flight_type`、`cabin`、`flights`、`passengers`、`payment_methods`、`total_baggages`、`nonfree_baggages`、`insurance` |
| `cancel_reservation` | `reservation_id` |
| `update_reservation_baggages` | `reservation_id`、`total_baggages`、`nonfree_baggages`、`payment_id` |
| `update_reservation_flights` | `reservation_id`、`cabin`、`flights`、`payment_id` |
| `update_reservation_passengers` | `reservation_id`、`passengers` |
| `send_certificate` | `user_id`、`amount` |

### Grounding 判定

当前实现只使用当前 action 之前的 tool response 抽取 `seen_entities`。模型自由文本中的值不视为可信来源。

write tool 参数中的以下字段会参与 grounding 检查：

```text
reservation_id
user_id
payment_id
flight_number
```

如果参数值没有出现在之前的 tool response 中，则该值被视为未 grounding。`payment_id` 和 `flight_number` 除了触发通用的 `prm/ungrounded_mutating_args_penalty`，还会分别触发更具体的扣分项。

### 安全调用序列奖励

`prm/safe_mutating_sequence_bonus=+0.08` 只在以下条件全部满足时触发：

1. 当前 action 是 write tool。
2. 当前 write tool 没有缺少必需参数。
3. 当前 action 之前至少调用过一个 read tool。
4. 对 `book_reservation`，满足上述条件即可继续检查；对其他 write tool，当前 action 之前还必须调用过 `get_reservation_details`。
5. 参数中的 `reservation_id`、`user_id`、`payment_id` 和 `flight_number` 都已 grounding。

read tool 包括：

```text
list_all_airports
search_direct_flight
search_onestop_flight
get_user_details
get_reservation_details
calculate
```

### 分项叠加示例

假设模型在没有任何 read tool 调用、没有确认、缺少 `payment_id` 的情况下，直接调用 `update_reservation_flights`，并填写了之前未由 tool response 返回的航班号。该 action 可以同时触发：

| 分项 | 分值 |
|---|---:|
| `prm/write_without_confirmation_penalty` | `-0.12` |
| `prm/premature_mutating_action_penalty` | `-0.12` |
| `prm/missing_required_mutating_arg_penalty` | 至少 `-0.08` |
| `prm/missing_payment_id_penalty` | `-0.20` |
| `prm/ungrounded_mutating_args_penalty` | `-0.12` |
| `prm/fabricated_flight_penalty` | `-0.18` |
| `legacy` P13：`update_reservation_flights` 使用未 grounding 航班号 | `-0.30` |

这些规则按条件叠加。随后整条轨迹仍按 `TAU_AIRLINE_PRM_NORMALIZE_MODE` 进行归一化，并在 `mean_clip` / `sum_clip` 模式下裁剪到 `[-0.5, 0.5]`。

### 日志指标

启用 `mutation_v1` 后，应在训练日志和 W&B 中看到以下指标：

```text
prm_rule_version_code/mean
prm/response_tool_mixing_penalty/mean
prm/ask_then_tool_penalty/mean
prm/confirm_then_write_without_wait_penalty/mean
prm/ask_then_wait_bonus/mean
prm/write_without_confirmation_penalty/mean
prm/ungrounded_mutating_args_penalty/mean
prm/fabricated_flight_penalty/mean
prm/ungrounded_payment_id_penalty/mean
prm/missing_required_mutating_arg_penalty/mean
prm/missing_payment_id_penalty/mean
prm/premature_mutating_action_penalty/mean
prm/certificate_policy_penalty/mean
prm/safe_mutating_sequence_bonus/mean
```

这些 `prm/*` 指标是轨迹中的原始分项累计值，便于定位行为变化；最终用于奖励的 `process_score` 仍会经过轨迹归一化和裁剪。

