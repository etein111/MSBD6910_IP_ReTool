"""冒烟测试前的静态预检 —— 不碰 GPU，在 login node 上跑。

验的是「tool 配置 → 类加载 → 实例化 → 真的执行代码」这条链，因为这条链
任何一环断了，冒烟作业都会在烧掉 GPU 配额之后才报错。

用法（在 code/work/ 下，已 activate retool 环境）：
    python preflight_check.py

全绿才提交冒烟作业。任一项 ✗ 都先修，别去占 GPU。
"""

import asyncio
import sys
import traceback

from omegaconf import OmegaConf
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.utils.tool_registry import get_tool_class

TOOL_CONFIG = "recipe/retool/local_tool_config.yaml"

_results: list[tuple[bool, str]] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    """记一条检查结果。detail 无论成败都打印，方便对照实际值。"""
    _results.append((ok, label))
    mark = "✓" if ok else "✗"
    print(f"  {mark} {label}" + (f"\n      → {detail}" if detail else ""))


def load_tool(timeout: float | None = None):
    """按 verl 的方式加载 tool 类并实例化。

    刻意复刻 tool_registry 里的调用形式（OmegaConf.to_container +
    model_validate），这样这里过了就说明 verl 那边也能过。
    """
    cfg = OmegaConf.load(TOOL_CONFIG)
    entry = cfg.tools[0]
    cls = get_tool_class(entry.class_name)
    schema = OpenAIFunctionToolSchema.model_validate(
        OmegaConf.to_container(entry.tool_schema, resolve=True)
    )
    config = OmegaConf.to_container(entry.config, resolve=True)
    if timeout is not None:
        config["default_timeout"] = timeout
    return cls(config=config, tool_schema=schema)


async def main() -> int:
    print("=" * 68)
    print("1. 配置与类加载")
    print("=" * 68)
    try:
        cfg = OmegaConf.load(TOOL_CONFIG)
        check(True, f"读到 {TOOL_CONFIG}", f"class_name={cfg.tools[0].class_name}")
        check(
            cfg.tools[0].tool_schema.function.name == "code_interpreter",
            "tool 名 = code_interpreter",
            f"实际={cfg.tools[0].tool_schema.function.name}",
        )
        tool = load_tool()
        check(True, "实例化 LocalCodeTool", f"python={tool.python}")
        check(isinstance(tool.python, str) and tool.python, "子进程解释器路径非空")
    except Exception:
        check(False, "配置/加载阶段抛异常")
        traceback.print_exc()
        return 1

    print()
    print("=" * 68)
    print("2. 执行行为（走完整 create → execute 契约）")
    print("=" * 68)
    iid, _ = await tool.create()

    async def run(code: str) -> str:
        resp, reward, metrics = await tool.execute(iid, {"code": code})
        # reward 必须恒为 None：ReTool 是 outcome-only，工具步不给分
        assert reward is None, f"exec_done 的 reward 应为 None，实际 {reward!r}"
        return resp.text or ""

    cases = [
        ("普通 print", "print(2 + 3)", "5"),
        ("自动补 print（裸表达式）", "2 + 3", "5"),
        ("多行 + 变量", "a = 6\nb = 7\nprint(a * b)", "42"),
        ("报错回原始 stderr", "1 / 0", "ZeroDivisionError"),
        ("Markdown 围栏剥离", "```python\nprint(7 * 6)\n```", "42"),
        ("模板标记净化", "print('<|im_end|>')", "< im_end >"),
        # --- 以下是 auto-print 的回归用例：官方朴素实现会在这几条上误伤 ---
        ("赋值结尾不补 print", "x = 1", ""),
        ("import 结尾不补 print", "import math", ""),
        ("缩进块结尾不补 print", "x = 2\nif x:\n    print(x)", "2"),
        ("循环结尾不补 print", "for i in range(3):\n    print(i)", "2"),
        ("已 print 不重复包", "print(123)", "123"),
        ("多行表达式结尾补 print", "total = 0\nfor i in range(4):\n    total += i\ntotal", "6"),
    ]
    for label, code, expect in cases:
        try:
            text = await run(code)
            if expect == "":
                check(text == "", label, f"返回={text!r}（期望空串）")
            else:
                check(expect in text, label, f"返回={text.strip()[:90]!r}")
        except Exception:
            check(False, label)
            traceback.print_exc()

    print()
    print("=" * 68)
    print("3. 沙箱隔离性")
    print("=" * 68)
    await run("leaked = 'i should not persist'")
    text = await run("print(leaked)")
    check(
        "NameError" in text,
        "每次执行无状态（上一轮的变量不可见）",
        f"返回={text.strip()[:90]!r}",
    )

    # 超时用单独一个短超时实例，别让整个预检等 30 秒
    print()
    print("=" * 68)
    print("4. 超时杀进程（用 3 秒超时的临时实例）")
    print("=" * 68)
    try:
        fast = load_tool(timeout=3)
        fid, _ = await fast.create()
        loop = asyncio.get_running_loop()
        start = loop.time()
        resp, _, _ = await fast.execute(fid, {"code": "while True: pass"})
        elapsed = loop.time() - start
        text = resp.text or ""
        check("timed out" in text.lower(), "死循环被杀", f"返回={text.strip()[:90]!r}")
        check(elapsed < 12, "没有无限挂起", f"耗时 {elapsed:.1f}s（超时设 3s）")
    except Exception:
        check(False, "超时路径抛异常")
        traceback.print_exc()

    print()
    print("=" * 68)
    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print(f"结果：{passed}/{total} 通过")
    failed = [label for ok, label in _results if not ok]
    if failed:
        print("未通过：")
        for label in failed:
            print(f"  ✗ {label}")
        print()
        print("→ 先修这些，别提交冒烟作业。")
        return 1
    print("→ 全绿，可以提交冒烟作业。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
