"""诊断 SFT 冷启动模型（global_step_372-hf）的多轮工具调用 rollout 轨迹。

忠实复现 verl 0.6.1 的 hermes 工具循环：
  - prompt = problem + answer_format（map_fn2 追加的 \\boxed 格式要求）
  - 用 tokenizer.apply_chat_template 构建（Qwen3 模板自带 <tool_call> 格式）
  - 生成后正则抽 <tool_call>{"name":...,"arguments":{...}}</tool_call>
  - 执行代码（复现 LocalCodeTool 的 fence 剥离 + auto-print + 子进程超时）
  - 回填 assistant(tool_calls) + tool(result)，继续生成，最多 8 用户轮

用法（GPU 节点，cwd=code/work）：
  python3 diag_sft_rollout.py
"""
import json
import os
import re
import signal
import subprocess
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/scratch/mscbdtspod2026/cyeah/retool/checkpoints/qwen3-1.7b-instruct-sft/global_step_372-hf"
DATA = "/scratch/mscbdtspod2026/cyeah/retool/data/dataset_clean/BytedTsinghua-SIA/DAPO-Math-17k-smoke/dapo-math-17k.parquet"
TOOL_CONFIG = "recipe/retool/local_tool_config.yaml"

# 命令行可覆盖模型路径 / 每轮 max_new_tokens：
#   python3 diag_sft_rollout.py [model_path] [max_new_tokens]
if len(sys.argv) > 1:
    MODEL = sys.argv[1]
MAX_NEW_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

answer_format = "\nThe answer format must be: \\boxed{'The final answer goes here.'}"
tool_call_re = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
boxed_re = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
code_fence_re = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)


def ensure_trailing_print(code: str) -> str:
    """复现 LocalCodeTool.ensure_trailing_print 的 auto-print 意图（简化 AST 版）。"""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code
    if isinstance(last.value, ast.Call) and getattr(last.value.func, "id", None) == "print":
        return code
    seg = ast.get_source_segment(code, last.value)
    if seg is None:
        return code
    lines = code.split("\n")
    head = lines[: last.lineno - 1]
    tail = lines[last.end_lineno:]
    return "\n".join([*head, f"print({seg})", *tail])


def run_code(code: str, timeout: int = 30) -> str:
    """复现 LocalCodeTool._run_code：子进程执行，超时杀组，返回 stdout/stderr。"""
    code = ensure_trailing_print(code)
    bootstrap = (
        "import resource\n"
        f"resource.setrlimit(resource.RLIMIT_CPU, ({timeout + 5}, {timeout + 5}))\n"
        f"exec(compile({code!r}, '<sandbox>', 'exec'))\n"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    try:
        p = subprocess.run(
            [sys.executable, "-B", "-c", bootstrap],
            capture_output=True, text=True, timeout=timeout, env=env,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout} seconds."
    if p.returncode == 0:
        return p.stdout
    return p.stderr if p.stderr else f"Error: process exited with code {p.returncode}."


def extract_final_answer(text: str):
    m = boxed_re.findall(text)
    return m[-1].strip() if m else None


def main():
    torch.manual_seed(0)
    print(f"=== 模型: {MODEL} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # 复现 verl init_class 的 tool_schemas 注入（tool_agent_loop.py:100,197,210）
    tools_config = OmegaConf.load(TOOL_CONFIG)
    tool_schema = OmegaConf.to_container(tools_config["tools"][0]["tool_schema"])
    tools = [tool_schema]
    print(f"=== 注入工具 schema: {tool_schema['function']['name']} ===")

    df = pd.read_parquet(DATA)
    print(f"=== 诊断数据 {len(df)} 行，跑 4 题 ===\n")

    for idx in [0, 5, 10, 20]:
        row = df.iloc[idx]
        problem = row["prompt"][0]["content"]
        gt = str(row["reward_model"]["ground_truth"])
        prompt = problem + answer_format
        print(f"{'=' * 80}")
        print(f"[题 {idx}]  ground_truth = {gt}")
        print(f"[题 {idx}]  problem 前 160 字 = {problem[:160].replace(chr(10), ' ')}")
        print("-" * 80)

        messages = [{"role": "user", "content": prompt}]
        final_answer = None

        for turn in range(8):
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", tools=tools
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_ids = out[0][input_ids.shape[1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=False)

            matches = tool_call_re.findall(text)
            reasoning = tool_call_re.sub("", text).strip()
            print(f"  --- turn {turn + 1} (生成 {len(new_ids)} token) ---")
            print(f"  [assistant 推理] {reasoning[:400]}{'…' if len(reasoning) > 400 else ''}")

            if not matches:
                # 无工具调用 → 视为最终答案
                final_answer = extract_final_answer(text)
                print(f"  [无工具调用，结束] 最终 \\boxed = {final_answer!r}")
                messages.append({"role": "assistant", "content": reasoning})
                break

            # 构造 assistant(tool_calls) 消息（对齐 SFT 数据格式：arguments 为 dict）
            assistant_msg = {"role": "assistant", "content": reasoning, "tool_calls": []}
            for m in matches:
                try:
                    fc = json.loads(m)
                except Exception as e:
                    print(f"  [!] tool_call JSON 解析失败: {e} | 原文: {m[:120]}")
                    continue
                assistant_msg["tool_calls"].append(
                    {"type": "function", "function": {"name": fc["name"], "arguments": fc["arguments"]}}
                )
            messages.append(assistant_msg)

            # 执行每个工具调用并回填 tool 消息
            for m in matches:
                try:
                    fc = json.loads(m)
                except Exception:
                    continue
                code = fc.get("arguments", {}).get("code", "")
                result = run_code(code)
                print(f"  [工具调用] {fc['name']}(code={code[:160]!r})")
                print(f"  [工具返回] {result[:300]!r}")
                messages.append({"role": "tool", "content": result})

        correct = (final_answer is not None and final_answer == gt)
        print(f"  >>> 判定: boxed={final_answer!r} gt={gt!r} → {'✅ 对' if correct else '❌ 错'}")
        print()

    print("=== 诊断结束 ===")


if __name__ == "__main__":
    main()
