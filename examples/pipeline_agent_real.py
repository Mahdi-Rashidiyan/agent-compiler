"""
examples/pipeline_agent_real.py
=================================
Agent 2 — QA Pipeline Agent (REAL LLM CALLS)

Three sequential steps on the same model/temperature → chain-merge eligible.

  draft_answer → refine_answer → format_answer

Pass 2 win: compiler detects the chain, collapses 3 API round-trips into 1
combined multi-part prompt. Saves 2 × network overhead.

  Unoptimised:  3 sequential calls  (~5–6s)
  Merged:       1 combined call     (~2–3s)  →  ~1.7× speedup
"""

from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType
from agentcompiler.backends.groq_backend import groq_call

MODEL    = "llama-3.3-70b-versatile"
QUESTION = "How does transformer self-attention work?"


async def draft_answer(ctx: dict) -> str:
    return await groq_call(
        f"Draft a technical answer to this question: '{QUESTION}' "
        f"Be concise. 2-3 sentences.",
        model=MODEL,
    )


async def refine_answer(ctx: dict) -> str:
    draft = ctx.get("draft_answer", "")
    return await groq_call(
        f"Improve this technical explanation for clarity and precision: '{draft}' "
        f"Fix any inaccuracies. 2-3 sentences.",
        model=MODEL,
    )


async def format_answer(ctx: dict) -> str:
    refined = ctx.get("refine_answer", "")
    return await groq_call(
        f"Format this into a final polished answer. "
        f"Add a one-line summary at the top, then the explanation: '{refined}'",
        model=MODEL,
        max_tokens=300,
    )


def build() -> ExecutionGraph:
    # All three share the same model + temperature → chain-merge eligible
    cfg = LLMConfig(model=MODEL, temperature=0.0, sim_latency_s=2.0)

    g = ExecutionGraph()
    g.add_node(Node("draft_answer",  NodeType.LLM_CALL, draft_answer,  llm_config=cfg))
    g.add_node(Node("refine_answer", NodeType.LLM_CALL, refine_answer,
                    dependencies=["draft_answer"], llm_config=cfg))
    g.add_node(Node("format_answer", NodeType.LLM_CALL, format_answer,
                    dependencies=["refine_answer"], llm_config=cfg))
    return g
