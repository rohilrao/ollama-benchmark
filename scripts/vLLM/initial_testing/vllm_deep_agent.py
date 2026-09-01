"""
Test: can ChatDeepSeek (pointed at your local Qwen3-235B vLLM server) be used
with LangChain's Deep Agents (`deepagents.create_deep_agent`) + a calculator
tool to solve a multi-step math problem requiring several sequential tool
calls?

Note: create_deep_agent is a more opinionated harness than plain
langchain.agents.create_agent — it bundles a default toolset (write_todos /
planning, filesystem read/write/edit, grep/glob, task delegation) and a
Claude-Code-style system prompt on top of your own tools. So you may see it
call write_todos before it ever touches the calculator — that's expected
deep-agent behavior, not a bug, and this script prints it so you can see it.

This streams the run and prints, as they happen:
  - model tokens as they're generated
  - every tool call the agent makes (name + args) -- including built-in
    deep-agent tools, not just your calculator
  - every tool result returned back to the agent
  - a final summary of every tool call made, in order

Install deps first:
    pip install --break-system-packages -U deepagents langchain-deepseek langchain-core

Usage:
    python deep_agent_calculator_streaming.py
"""

import ast
import math
import operator
import os

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_deepseek import ChatDeepSeek

# ---- Config: point this at your podman-hosted vLLM/OpenAI-compatible server ----
BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("QWEN_API_KEY", "EMPTY")
MODEL = os.environ.get("QWEN_MODEL", "qwen3-235b")

PROBLEM = (
    "A rectangular garden is 24 meters by 15 meters. A second garden is a "
    "square with the exact same area as the rectangular one. I want to put "
    "fencing around BOTH gardens, and fencing costs $12.50 per meter. "
    "What is the total fencing cost? Use the calculator tool for every "
    "individual arithmetic step — don't compute anything in your head."
)


# ---------------------------------------------------------------------------
# A safe calculator tool (no raw eval() — restricted AST arithmetic only)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        fn = _FUNCS[node.func.id]
        return fn(*[_eval_node(a) for a in node.args])
    raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.
    Supports + - * / ** and the functions sqrt(), abs(), round().
    Example inputs: '24 * 15', 'sqrt(360)', '2 * (24 + 15)'.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Build the model + deep agent
# ---------------------------------------------------------------------------
llm = ChatDeepSeek(
    api_base=BASE_URL,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)

agent = create_deep_agent(
    model=llm,
    tools=[calculator],
    system_prompt=(
        "You are a careful math assistant. Use the calculator tool for "
        "every individual arithmetic operation — never compute arithmetic "
        "yourself. Keep planning minimal; this is a small, well-defined "
        "math problem, not a research task."
    ),
)


def main():
    print(f"Target server: {BASE_URL}  |  model: {MODEL}")
    print(f"\nProblem: {PROBLEM}\n")
    print("=" * 70)
    print("STREAMING RUN (deepagents.create_deep_agent)")
    print("=" * 70)

    tool_call_log = []    # (name, args) in call order
    tool_result_log = []  # (name, result) in return order

    for stream_mode, chunk in agent.stream(
        {"messages": [{"role": "user", "content": PROBLEM}]},
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "messages":
            message_chunk, metadata = chunk
            # Token-by-token text from the model as it's generated
            if isinstance(message_chunk, AIMessageChunk) and message_chunk.content:
                print(message_chunk.content, end="", flush=True)

        elif stream_mode == "updates":
            for node_name, update in chunk.items():
                messages = update.get("messages", []) if isinstance(update, dict) else []
                for msg in messages:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"\n\n[TOOL CALL]   node={node_name}  {tc['name']}({tc['args']})")
                            tool_call_log.append((tc["name"], tc["args"]))
                    elif isinstance(msg, ToolMessage):
                        # Truncate long results (e.g. filesystem/todo tool output) for readability
                        content = str(msg.content)
                        shown = content if len(content) < 300 else content[:300] + " ...[truncated]"
                        print(f"[TOOL RESULT] node={node_name}  {msg.name} -> {shown}\n")
                        tool_result_log.append((msg.name, content))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total tool calls made: {len(tool_call_log)}\n")
    for i, ((name, args), (_, result)) in enumerate(zip(tool_call_log, tool_result_log), 1):
        shown = result if len(result) < 200 else result[:200] + " ...[truncated]"
        print(f"  {i}. {name}({args}) -> {shown}")

    calc_calls = [c for c in tool_call_log if c[0] == "calculator"]
    print(f"\ncalculator tool specifically was called {len(calc_calls)} time(s).")


if __name__ == "__main__":
    main()
