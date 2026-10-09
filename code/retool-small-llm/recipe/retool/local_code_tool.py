"""本地 subprocess 代码解释器 Tool —— 给 verl 多轮 rollout 用。

为什么不用官方 SandboxFusion
============================
官方 recipe 的 SandboxFusionTool 依赖一个独立部署的 HTTP 服务
（sandbox_fusion_url，通常 localhost:8080，需要 Docker）。SuperPOD 的
计算节点不一定允许跑 Docker，且 Gate 1 只想先打通链路。

本实现把 agentic-rl-lab/05-retool 里 LocalPythonSandbox 的设计搬进
verl 的 BaseTool 接口：纯 Python、零外部依赖、无需 Docker。

执行模型
========
模型给出的 Python 代码用 `python -B -c <bootstrap>` 起全新子进程执行。
list 形式 argv 不经过 shell，代码经 repr() 内嵌进引导串——不落脚本文件、
无注入面；每次调用一个全新进程，满足"每次执行独立无状态"的语义。

四道保险（与 05-retool 一致）
============================
1. 限并发：threading.BoundedSemaphore。不用 asyncio.Semaphore，因为它会
   绑定到单个事件循环，而训练循环会跨 loop 复用。
2. 限时间：wall-clock 超时后 os.killpg 杀死整个进程组
   （start_new_session=True 才能整组杀）；子进程内 RLIMIT_CPU 兜底。
3. 限输出：stdout/stderr 写匿名临时文件而非 PIPE，防止 print 洪水撑爆
   父进程内存；只读文件尾部并截断。
4. 限线程：子进程环境把 BLAS 线程数压到 1，防止并发沙箱打满机器。

结果哲学
========
成功回 stdout（无 print 则为空串）；失败回原始 stderr（完整 traceback）；
超时回超时说明。报错原样回喂——论文中"自我纠错"正是从看到错误反馈
再改代码中涌现的。唯一处理：模板特殊标记替换成无害写法，防止污染
observation 结构。
"""

import ast
import asyncio
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op


# 子进程引导代码：先限制 CPU 时间，再 exec 模型代码。
# repr 内嵌避免转义问题；exec 保持 __main__ 语义。
_BOOTSTRAP = (
    "import resource\n"
    "resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))\n"
    "exec(compile({code}, '<sandbox>', 'exec'))\n"
)

# 子进程环境：保留基础环境，但把数值库线程数压到 1，
# 防止多个并发沙箱各自拉起一整组 BLAS 线程把机器打满。
_CHILD_ENV_OVERRIDES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

# 工具返回文本里的模板特殊标记会污染 observation 结构，统一替换成无害写法。
_SANITIZE_REPLACEMENTS = {
    "<|im_start|>": "< im_start >",
    "<|im_end|>": "< im_end >",
    "<tool_response>": "< tool_response >",
    "</tool_response>": "< /tool_response>",
    "<tool_call>": "< tool_call >",
    "</tool_call>": "< /tool_call>",
}


def sanitize_tool_content(text: str) -> str:
    """把 tool 返回文本中的模板特殊标记替换成无害写法。"""
    for marker, replacement in _SANITIZE_REPLACEMENTS.items():
        text = text.replace(marker, replacement)
    return text


def ensure_trailing_print(code: str) -> str:
    """给末尾的裸表达式补 print() —— 官方 auto-print 意图的精确实现。

    模型经常写 `2 + 3` 而不写 `print(2 + 3)`，不补就会得到空输出。

    与官方 recipe 的差异（**有意为之，非笔误**）
    --------------------------------------------
    官方 `retool.py` 是无条件把最后一行非空行包进 print()：

        if not lines[i].startswith("print"):
            lines[i] = f"print({line})"

    这对「以表达式结尾」的代码是对的，但会误伤两类极常见的写法：

      1. 以语句结尾：`x = 1`  → `print(x = 1)` → TypeError
                     `while True: pass` → `print(while True: pass)` → SyntaxError
      2. 以缩进块结尾：`if x:\\n    print(x)` 的最后一行是 `    print(x)`，
         不以 `print` 开头 → `print(    print(x))` → SyntaxError

    误伤的后果不是「少一次输出」，而是**把一段假的 traceback 喂回给模型**，
    污染 observation 和 reward 信号。官方没炸是因为它跑的是 SFT 冷启动后的
    模型（被训练成总以 `print(...)` 收尾），这条路径几乎不触发；而本项目
    的 H1 假设正是「基座小模型不会写 print」，命中率极高。

    所以这里按官方注释写明的**意图**（"add a print statement to the end"）
    用 AST 精确判断：只有当最后一条**顶层语句是裸表达式**、且不是 print
    调用时，才补 print。语义与官方一致，只是不再误伤非表达式语句。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 语法错误原样交给沙箱去报，别在这里二次加工成更难懂的报错
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code  # 赋值 / import / def / for / if / while ... 不是裸表达式
    value = last.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "print"
    ):
        return code  # 已经是 print 调用，别套两层
    source = ast.get_source_segment(code, value)
    if source is None:
        return code
    lines = code.split("\n")
    # lineno/end_lineno 是 1-based 且含端点，据此替换该语句占用的整段
    head = lines[: last.lineno - 1]
    tail = lines[last.end_lineno :]
    return "\n".join([*head, f"print({source})", *tail])


class LocalCodeTool(BaseTool):
    """用本机 Python 解释器执行模型生成的代码，带超时和资源限制。

    config 可选项：
      default_timeout  单次执行 wall-clock 上限（秒），默认 30
      num_workers      并发沙箱上限，默认 8
      max_output_chars stdout/stderr 各自保留的最大字符数，默认 4096
      python           子进程使用的解释器，默认当前环境的 sys.executable
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.default_timeout = float(config.get("default_timeout", 30))
        self.max_workers = int(config.get("num_workers", 8))
        self.max_output_chars = int(config.get("max_output_chars", 4096))
        self.python = config.get("python", sys.executable)
        # 线程信号量而非 asyncio.Semaphore：后者绑定单一事件循环，跨 loop 复用会报错。
        self._semaphore = threading.BoundedSemaphore(self.max_workers)
        # 模型有时把代码包在 ```python 围栏里，剥掉外层围栏。
        self.code_pattern = re.compile(r"```(?:python)?(.*?)```", re.DOTALL)

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        """为一条轨迹创建工具实例（本沙箱无状态，仅分配 id）。"""
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """执行模型给出的代码，返回 (ToolResponse, reward, metrics)。

        reward 恒为 None：ReTool 是 outcome-only 奖励，调用工具本身不给分。
        """
        code = parameters.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()
        code = ensure_trailing_print(code)
        # 阻塞式子进程执行放到线程池，避免卡住 rollout 的事件循环。
        text = await asyncio.to_thread(self._run_code, code)
        return ToolResponse(text=text), None, None

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """ReTool 的奖励只来自最终答案对不对，工具步不单独给分。"""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """本沙箱每次执行都是全新进程，无状态可释放。"""
        pass

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _read_capped(self, handle: Any, limit: int) -> str:
        """只从输出文件尾部读取有限字节，避免 print 洪水撑爆内存。"""
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")

    def _truncate_tail(self, text: str) -> str:
        """按尾部保留输出；模型的关键结果通常打在末尾。"""
        if len(text) <= self.max_output_chars:
            return text
        omitted = len(text) - self.max_output_chars
        return f"[... truncated {omitted} chars ...]\n{text[-self.max_output_chars:]}"

    def _run_code(self, code: str) -> str:
        """同步执行一段代码，返回喂给模型的 tool 文本。"""
        bootstrap = _BOOTSTRAP.format(
            cpu=int(self.default_timeout) + 5, code=repr(code)
        )
        env = {**os.environ, **_CHILD_ENV_OVERRIDES}
        with self._semaphore:
            # 输出写匿名临时文件而非 PIPE，防止子进程输出洪水撑爆父进程内存。
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    [self.python, "-B", "-c", bootstrap],
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=env,
                    start_new_session=True,  # 独立进程组，超时才能整组杀死
                )
                timed_out = False
                try:
                    returncode = process.wait(timeout=self.default_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = None
                    process.wait()  # 回收僵尸进程
                stdout = self._truncate_tail(
                    self._read_capped(stdout_file, self.max_output_chars * 4)
                )
                stderr = self._truncate_tail(
                    self._read_capped(stderr_file, self.max_output_chars * 4)
                )

        if timed_out:
            return f"Error: execution timed out after {self.default_timeout:.0f} seconds."
        if returncode == 0:
            return sanitize_tool_content(stdout)  # 无 print 时为空字符串（官方语义）
        if stderr:
            return sanitize_tool_content(stderr)
        return f"Error: process exited with code {returncode}."
