"""
benchmarks/benchmark.py
========================
Measures real wall-clock speedups for all three agents.

LLM calls are simulated with asyncio.sleep() so latency values are
artificial, but CONCURRENCY IS REAL — asyncio.gather() genuinely
overlaps coroutines.  Speedup ratios reflect true parallel scheduling.

Run:
    cd agentcompiler
    python -m benchmarks.benchmark
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Callable, Dict, Tuple

sys.path.insert(0, ".")

from agentcompiler.compiler import AgentCompiler
from agentcompiler.graph import ExecutionGraph, Node, NodeType, LLMConfig
from agentcompiler.passes.parallelism import ParallelismExtractionPass
from agentcompiler.passes.merging import LLMCallMergingPass
from agentcompiler.passes.speculative import SpeculativeBranchPass
from agentcompiler.runtime.executor import AgentExecutor

import examples.research_agent  as research
import examples.pipeline_agent  as pipeline
import examples.branching_agent as branching


async def timed_run(graph: ExecutionGraph) -> Tuple[float, Dict]:
    executor = AgentExecutor()
    t0 = time.perf_counter()
    result = await executor.run(graph)
    return time.perf_counter() - t0, result


async def run_pair(builder: Callable, passes) -> Tuple[float, float]:
    t_raw, _ = await timed_run(builder())
    compiler  = AgentCompiler(passes=passes)
    opt_graph = compiler.compile(builder())
    t_opt, _ = await timed_run(opt_graph)
    return t_raw, t_opt


def _combined_graph() -> ExecutionGraph:
    import asyncio as _a
    g = ExecutionGraph()

    q_cfg = LLMConfig("claude-3-haiku", 0.2, 512, 1.0)
    a_cfg = LLMConfig("claude-3-haiku", 0.0, 512, 0.9)
    c_cfg = LLMConfig("claude-3-haiku", 0.0, 512, 0.5)
    d_cfg = LLMConfig("claude-3-haiku", 0.7, 512, 1.2)
    f_cfg = LLMConfig("claude-3-haiku", 0.3, 512, 0.4)

    async def q1(ctx):  await _a.sleep(1.0); return "data-A"
    async def q2(ctx):  await _a.sleep(1.0); return "data-B"
    async def q3(ctx):  await _a.sleep(1.0); return "data-C"
    async def analyse(ctx): await _a.sleep(0.9); return "analysis-1"
    async def deepen(ctx):  await _a.sleep(0.9); return "analysis-2"
    async def classify(ctx): await _a.sleep(0.5); return True
    async def pos_resp(ctx): await _a.sleep(1.2); return "positive"
    async def neg_resp(ctx): await _a.sleep(1.2); return "negative"
    async def final(ctx):   await _a.sleep(0.4); return "done"

    g.add_node(Node("q1", NodeType.LLM_CALL, q1, llm_config=q_cfg))
    g.add_node(Node("q2", NodeType.LLM_CALL, q2, llm_config=q_cfg))
    g.add_node(Node("q3", NodeType.LLM_CALL, q3, llm_config=q_cfg))
    g.add_node(Node("analyse", NodeType.LLM_CALL, analyse,
                    dependencies=["q1","q2","q3"], llm_config=a_cfg))
    g.add_node(Node("deepen", NodeType.LLM_CALL, deepen,
                    dependencies=["analyse"], llm_config=a_cfg))
    g.add_node(Node("classify", NodeType.CONDITION, classify,
                    dependencies=["deepen"], llm_config=c_cfg,
                    true_branch="pos_resp", false_branch="neg_resp", p_true=0.8))
    g.add_node(Node("pos_resp", NodeType.LLM_CALL, pos_resp,
                    llm_config=d_cfg, metadata={"condition_branch": "classify"}))
    g.add_node(Node("neg_resp", NodeType.LLM_CALL, neg_resp,
                    llm_config=d_cfg, metadata={"condition_branch": "classify"}))
    g.add_node(Node("final", NodeType.LLM_CALL, final,
                    dependencies=["classify"], llm_config=f_cfg))
    return g


BENCHMARKS = [
    {
        "name":    "Agent 1 — Research   (3 independent queries + synthesis)",
        "builder": research.build,
        "passes":  [ParallelismExtractionPass()],
        "label":   "Pass 1 · Parallelism",
        "theory":  "3.80s → 1.80s  (2.1×)",
    },
    {
        "name":    "Agent 2 — QA Pipeline (3 sequential calls, same model)",
        "builder": pipeline.build,
        "passes":  [LLMCallMergingPass()],
        "label":   "Pass 2 · LLM Merging",
        "theory":  "3.00s → 2.40s  (1.25×)",
    },
    {
        "name":    "Agent 3 — Branch      (speculative execution, P=0.80)",
        "builder": branching.build,
        "passes":  [SpeculativeBranchPass()],
        "label":   "Pass 3 · Speculative",
        "theory":  "2.00s → 1.50s  (1.33×)",
    },
    {
        "name":    "Combined              (all three passes)",
        "builder": _combined_graph,
        "passes":  [ParallelismExtractionPass(), LLMCallMergingPass(), SpeculativeBranchPass()],
        "label":   "All 3 passes",
        "theory":  "6.90s → 4.10s  (1.68×)",
    },
]

DIV = "─" * 74


async def main() -> None:
    print()
    print("═" * 74)
    print("  ML Agent Compiler — Benchmark Suite  v0.1.0")
    print("  Latency: simulated via asyncio.sleep()   Concurrency: real")
    print("═" * 74)

    rows = []
    for b in BENCHMARKS:
        print(f"\n▶  {b['name']}")
        print(f"   Pass: {b['label']}   (theoretical: {b['theory']})")
        t_raw, t_opt = await run_pair(b["builder"], b["passes"])
        speedup = t_raw / t_opt
        rows.append((b["name"], b["label"], t_raw, t_opt, speedup))
        print(f"   Unoptimised : {t_raw:.3f} s")
        print(f"   Optimised   : {t_opt:.3f} s")
        print(f"   Speedup     : {speedup:.2f}×")

    print()
    print(DIV)
    print(f"  {'Agent':<42} {'Pass':<22} {'Raw':>6} {'Opt':>6}  {'×':>6}")
    print(DIV)
    for name, label, raw, opt, sp in rows:
        print(f"  {name[:42]:<42} {label:<22} {raw:>5.2f}s {opt:>5.2f}s  {sp:>5.2f}×")
    print(DIV)
    avg = sum(r[4] for r in rows) / len(rows)
    print(f"  {'Average speedup across all benchmarks':<68} {avg:>5.2f}×")
    print(DIV)
    print()


if __name__ == "__main__":
    asyncio.run(main())
