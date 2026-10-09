#!/usr/bin/env python3
"""补丁：让 verl 0.6.1 把 num_turns 传进 extra_fields（→ extra_info → compute_score）。

根因：verl-recipe/retool/retool.py 的 compute_score 读 extra_info["num_turns"]，
但 verl 0.6.1 把轮数存为 AgentLoopOutput.num_turns → non_tensor_batch 的 __num_turns__，
并没有放进 extra_info（extra_fields 里只有 turn_scores/tool_rewards），导致 KeyError: 'num_turns'。

修法：在 tool_agent_loop.py 构造 AgentLoopOutput 时，把 num_turns 一并塞进 extra_fields。
幂等：若已含 "num_turns" 键则跳过。
"""
import sys

F = "/home/cyeah/.conda/envs/retool/lib/python3.11/site-packages/verl/experimental/agent_loop/tool_agent_loop.py"

OLD = '        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})'
NEW = (
    '        output.extra_fields.update(\n'
    '            {\n'
    '                "turn_scores": agent_data.turn_scores,\n'
    '                "tool_rewards": agent_data.tool_rewards,\n'
    '                "num_turns": agent_data.user_turns + agent_data.assistant_turns + 1,\n'
    '            }\n'
    '        )'
)

s = open(F).read()

if '"num_turns"' in s and 'agent_data.user_turns + agent_data.assistant_turns + 1' in s:
    print("SKIP: 已打过补丁")
    sys.exit(0)

if s.count(OLD) != 1:
    print(f"ERROR: 未找到唯一的待替换串（count={s.count(OLD)}），文件可能已变，需手动检查 {F}")
    sys.exit(1)

s = s.replace(OLD, NEW)
open(F, "w").write(s)
print("PATCH_OK: 已注入 num_turns 到 extra_fields")
