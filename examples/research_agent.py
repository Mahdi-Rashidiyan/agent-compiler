"""
examples/research_agent.py
============================
Agent 1 — Research Agent

Fetches information on three independent topics, then synthesises.

Graph topology:
  query_climate ─┐
  query_energy  ─┤─→ synthesize
  query_policy  ─┘

Pass 1 win: three queries have no inter-dependency → run concurrently.
  Unoptimised:  3×1.0s + 0.8s = 3.8s
  Optimised:    1.0s   + 0.8s = 1.8s   →  ~2.1× speedup
"""

import asyncio
from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType

QUERY_LAT = 1.0
SYNTH_LAT = 0.8


async def query_climate(ctx: dict) -> str:
    await asyncio.sleep(QUERY_LAT)
    return "CO2 levels at 421 ppm; renewables now 30% of global electricity"


async def query_energy(ctx: dict) -> str:
    await asyncio.sleep(QUERY_LAT)
    return "Solar LCOE fell 89% in a decade; wind at 8.3 ¢/kWh globally"


async def query_policy(ctx: dict) -> str:
    await asyncio.sleep(QUERY_LAT)
    return "195 nations under Paris Agreement; 68 have carbon-pricing mechanisms"


async def synthesize(ctx: dict) -> str:
    await asyncio.sleep(SYNTH_LAT)
    climate = ctx.get("query_climate", "")
    energy  = ctx.get("query_energy",  "")
    policy  = ctx.get("query_policy",  "")
    return f"Summary — Climate: {climate}. Energy: {energy}. Policy: {policy}."


def build() -> ExecutionGraph:
    q_cfg = LLMConfig(model="claude-3-haiku", temperature=0.2, sim_latency_s=QUERY_LAT)
    s_cfg = LLMConfig(model="claude-3-haiku", temperature=0.3, sim_latency_s=SYNTH_LAT)

    g = ExecutionGraph()
    g.add_node(Node("query_climate", NodeType.LLM_CALL, query_climate, llm_config=q_cfg))
    g.add_node(Node("query_energy",  NodeType.LLM_CALL, query_energy,  llm_config=q_cfg))
    g.add_node(Node("query_policy",  NodeType.LLM_CALL, query_policy,  llm_config=q_cfg))
    g.add_node(Node("synthesize",    NodeType.LLM_CALL, synthesize,    llm_config=s_cfg))
    g.add_edge("query_climate", "synthesize")
    g.add_edge("query_energy",  "synthesize")
    g.add_edge("query_policy",  "synthesize")
    return g
