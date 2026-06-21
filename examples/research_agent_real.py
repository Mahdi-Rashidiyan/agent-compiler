"""
examples/research_agent_real.py
================================
Agent 1 — Research Agent (REAL LLM CALLS)

Three independent topic queries run via Groq, then synthesised.

Graph:
  query_climate ─┐
  query_energy  ─┤─→ synthesize
  query_policy  ─┘

Pass 1 win: all three queries are independent.
  Unoptimised:  3 sequential Groq calls  (~6–8s total)
  Optimised:    3 concurrent Groq calls  (~2–3s total)  →  ~2.5× speedup
"""

from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType
from agentcompiler.backends.groq_backend import groq_call

MODEL = "llama-3.3-70b-versatile"


async def query_climate(ctx: dict) -> str:
    return await groq_call(
        "What are the three most significant climate change developments in 2024-2025? "
        "Be precise and concise. Maximum 2 sentences.",
        model=MODEL,
    )


async def query_energy(ctx: dict) -> str:
    return await groq_call(
        "What are the key global renewable energy milestones and trends in 2024-2025? "
        "Be precise and concise. Maximum 2 sentences.",
        model=MODEL,
    )


async def query_policy(ctx: dict) -> str:
    return await groq_call(
        "What are the most important international climate policy agreements or frameworks "
        "active in 2024-2025? Be precise and concise. Maximum 2 sentences.",
        model=MODEL,
    )


async def synthesize(ctx: dict) -> str:
    climate = ctx.get("query_climate", "")
    energy  = ctx.get("query_energy",  "")
    policy  = ctx.get("query_policy",  "")

    return await groq_call(
        f"Synthesize these three research findings into one coherent paragraph (3-4 sentences):\n\n"
        f"Climate findings: {climate}\n"
        f"Energy findings:  {energy}\n"
        f"Policy findings:  {policy}\n\n"
        f"Synthesis:",
        model=MODEL,
        max_tokens=300,
        temperature=0.3,
    )


def build() -> ExecutionGraph:
    cfg      = LLMConfig(model=MODEL, temperature=0.0,  sim_latency_s=2.0)
    synth    = LLMConfig(model=MODEL, temperature=0.3,  sim_latency_s=2.0)

    g = ExecutionGraph()
    g.add_node(Node("query_climate", NodeType.LLM_CALL, query_climate, llm_config=cfg))
    g.add_node(Node("query_energy",  NodeType.LLM_CALL, query_energy,  llm_config=cfg))
    g.add_node(Node("query_policy",  NodeType.LLM_CALL, query_policy,  llm_config=cfg))
    g.add_node(Node("synthesize",    NodeType.LLM_CALL, synthesize,
                    dependencies=["query_climate", "query_energy", "query_policy"],
                    llm_config=synth))
    return g
