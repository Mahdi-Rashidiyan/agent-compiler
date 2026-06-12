"""
examples/pipeline_agent.py
============================
Agent 2 — QA Pipeline Agent

Produces a polished answer via three sequential steps that share the same
model/temperature (merge-eligible):

  draft_answer → refine_answer → format_answer → (done)

Each step depends on the previous: naive execution = 3 sequential API calls.

Pass 2 win: compiler detects the chain, reformulates as one multi-part
prompt, eliminating 2 API round-trip overheads.

  Unoptimised: 3 × (0.3 API_OH + 0.7 gen) = 3.0 s
  Merged:      0.3 API_OH + 3 × 0.7 gen   = 2.4 s   (chain-merge)
             — but with batch efficiency (0.85×): ≈ 2.1 s → ~1.4× speedup

The real gain is even larger under rate-limiting (1 request instead of 3).
"""

import asyncio
from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType

STEP_LAT = 1.0   # 0.3s API overhead + 0.7s generation


async def draft_answer(ctx: dict) -> str:
    await asyncio.sleep(STEP_LAT)
    return "Draft: The transformer architecture uses self-attention to process sequences."


async def refine_answer(ctx: dict) -> str:
    await asyncio.sleep(STEP_LAT)
    draft = ctx.get("draft_answer", "")
    return f"Refined: {draft} Multi-head attention allows parallel focus on subspaces."


async def format_answer(ctx: dict) -> str:
    await asyncio.sleep(STEP_LAT)
    refined = ctx.get("refine_answer", "")
    return f"**Final Answer**\n{refined}\n\nSee Vaswani et al. 2017 for details."


def build() -> ExecutionGraph:
    cfg = LLMConfig(model="claude-3-haiku", temperature=0.0, sim_latency_s=STEP_LAT)

    g = ExecutionGraph()
    g.add_node(Node("draft_answer",  NodeType.LLM_CALL, draft_answer,  llm_config=cfg))
    g.add_node(Node("refine_answer", NodeType.LLM_CALL, refine_answer, llm_config=cfg))
    g.add_node(Node("format_answer", NodeType.LLM_CALL, format_answer, llm_config=cfg))
    g.add_edge("draft_answer",  "refine_answer")
    g.add_edge("refine_answer", "format_answer")
    return g
