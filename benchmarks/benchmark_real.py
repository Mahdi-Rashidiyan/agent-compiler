"""
benchmarks/benchmark_real.py
==============================
REAL benchmark — actual Groq API calls, actual wall-clock timing.

This is the number you show to VCs and put in the README.

Setup:
    pip install groq
    export GROQ_API_KEY="your_key_here"

Run:
    cd agentcompiler
    python -m benchmarks.benchmark_real
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Callable, Dict, Tuple

sys.path.insert(0, ".")

from agentcompiler.compiler import AgentCompiler
from agentcompiler.graph import ExecutionGraph
from agentcompiler.passes.parallelism import ParallelismExtractionPass
from agentcompiler.passes.merging import LLMCallMergingPass
from agentcompiler.passes.speculative import SpeculativeBranchPass
from agentcompiler.runtime.executor import AgentExecutor

import examples.research_agent_real  as research
import examples.pipeline_agent_real  as pipeline
import examples.branching_agent_real as branching


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def timed_run(graph: ExecutionGraph) -> Tuple[float, Dict]:
    executor = AgentExecutor()
    t0 = time.perf_counter()
    result = await executor.run(graph)
    return time.perf_counter() - t0, result


async def run_pair(
    builder: Callable,
    passes,
    cooldown: float = 10.0,
) -> Tuple[float, float, Dict, Dict]:
    """
    Run unoptimised then optimised, with a cooldown between runs
    to avoid Groq rate limits.
    """
    print(f"    Running unoptimised...", flush=True)
    t_raw, raw_result = await timed_run(builder())

    print(f"    Cooling down {cooldown}s (rate limit buffer)...", flush=True)
    await asyncio.sleep(cooldown)

    compiler  = AgentCompiler(passes=passes)
    opt_graph = compiler.compile(builder())

    print(f"    Running optimised...", flush=True)
    t_opt, opt_result = await timed_run(opt_graph)

    return t_raw, t_opt, raw_result, opt_result


# ─────────────────────────────────────────────────────────────────────────────
# Benchmarks
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARKS = [
    {
        "name":       "Agent 1 — Research   (3 independent queries + synthesis)",
        "builder":    research.build,
        "passes":     [ParallelismExtractionPass()],
        "label":      "Pass 1 · Parallelism",
        "result_key": "synthesize",
    },
    {
        "name":       "Agent 2 — QA Pipeline (3 sequential calls, chain merge)",
        "builder":    pipeline.build,
        "passes":     [LLMCallMergingPass()],
        "label":      "Pass 2 · LLM Merging",
        "result_key": "format_answer",
    },
    {
        "name":       "Agent 3 — Branch      (speculative execution, P=0.80)",
        "builder":    branching.build,
        "passes":     [SpeculativeBranchPass()],
        "label":      "Pass 3 · Speculative",
        "result_key": "compose_reply",
    },
]

DIV = "─" * 74


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print()
    print("═" * 74)
    print("  Tochal — Real Benchmark  (Groq · llama-3.3-70b-versatile)")
    print("  Concurrency: real asyncio   LLM calls: real Groq API")
    print("═" * 74)

    rows = []

    for b in BENCHMARKS:
        print(f"\n▶  {b['name']}")
        print(f"   Pass: {b['label']}")

        t_raw, t_opt, raw_res, opt_res = await run_pair(b["builder"], b["passes"])
        speedup = t_raw / t_opt if t_opt > 0 else 0.0
        rows.append((b["name"], b["label"], t_raw, t_opt, speedup))

        print(f"\n   Unoptimised : {t_raw:.2f}s")
        print(f"   Optimised   : {t_opt:.2f}s")
        print(f"   Speedup     : {speedup:.2f}×")

        # Show a snippet of the real LLM output
        key    = b["result_key"]
        output = opt_res.get(key, "")
        if output:
            snippet = output[:200] + ("..." if len(output) > 200 else "")
            print(f"\n   Sample output ({key}):")
            print(f"   {snippet}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(DIV)
    print(f"  {'Agent':<42} {'Pass':<22} {'Raw':>6} {'Opt':>6}  {'×':>6}")
    print(DIV)
    for name, label, raw, opt, sp in rows:
        print(f"  {name[:42]:<42} {label:<22} {raw:>5.2f}s {opt:>5.2f}s  {sp:>5.2f}×")
    print(DIV)
    if rows:
        avg = sum(r[4] for r in rows) / len(rows)
        print(f"  {'Average speedup (real Groq API)':<68} {avg:>5.2f}×")
    print(DIV)
    print()
    print("  Model:   llama-3.3-70b-versatile  via  api.groq.com")
    print("  These numbers are real. Copy them into your README.")
    print()


if __name__ == "__main__":
    asyncio.run(main())
