# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""字段映射 map_fn/map_fn2 —— 相对官方 verl-recipe/retool 的改动：

DAPO-Math-17k 的题目外面统一包了一层模板，要求用 "Answer: $Answer" 格式作答，
与 answer_format 追加的 \\boxed{} 格式（reward `strict_box_verify=True` 只认 boxed）冲突。
map_fn2 现在先把 DAPO 模板的 prefix/suffix 剥掉，只留题目本体，再追加 \\boxed{} 要求，
保证 prompt 内只有一套格式指令。AIME 的 map_fn 不变（AIME 无该模板）。
"""

answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

# DAPO-Math-17k 外层模板（与 agentic-rl-lab/05-retool/prepare_data.py 保持一致）
DAPO_PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
DAPO_PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def _strip_dapo_template(content: str) -> str:
    """剥掉 DAPO 统一外层模板，只保留题目本体；模板不完整时原样保留。"""
    if content.startswith(DAPO_PROMPT_PREFIX):
        content = content[len(DAPO_PROMPT_PREFIX) :]
    if content.endswith(DAPO_PROMPT_SUFFIX):
        content = content[: -len(DAPO_PROMPT_SUFFIX)]
    return content.strip()


def map_fn(row: dict, *, data_source: str = None):
    if data_source == "Maxwell-Jia/AIME_2024":
        problem, answer = row["Problem"], row["Answer"]
    elif data_source == "yentinglin/aime_2025":
        problem, answer = row["problem"], row["answer"]
    prompt = problem + answer_format
    data = {
        "data_source": data_source.split("/")[1].lower(),  # aime_2024, aime_2025
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "MATH",
        "reward_model": {"ground_truth": str(answer)},
        "agent_name": "tool_agent",
    }
    return data


def map_fn2(row: dict):
    content = row["prompt"][0]["content"]
    row["prompt"][0]["content"] = _strip_dapo_template(content) + answer_format
    row["agent_name"] = "tool_agent"
    return row
