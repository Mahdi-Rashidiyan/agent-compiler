"""
examples/branching_agent_real.py
==================================
Agent 3 — Sentiment Branch Agent (REAL LLM CALLS)

Classifies customer feedback → routes to positive or negative response.
80% of real customer feedback is positive → speculative execution fires.

Pass 3 win: draft_positive pre-starts concurrently with classify_sentiment.
  Unoptimised:  classify (2s) → draft (2s) → compose (1s) = ~5s
  Optimised:    [classify ∥ draft_positive] = max(2,2) = 2s → compose 1s = ~3s
  Speedup:      ~1.5×
"""

from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType
from agentcompiler.backends.groq_backend import groq_call

MODEL    = "llama-3.3-70b-versatile"
FEEDBACK = (
    "The product was absolutely outstanding. "
    "Setup was seamless, performance exceeded all expectations. "
    "Genuinely one of the best purchases I have made this year."
)


async def classify_sentiment(ctx: dict) -> bool:
    result = await groq_call(
        f"Classify the sentiment of this customer feedback. "
        f"Reply with ONLY one word — either 'positive' or 'negative', nothing else.\n\n"
        f"Feedback: '{FEEDBACK}'",
        model=MODEL,
        max_tokens=5,
    )
    return "positive" in result.strip().lower()


async def draft_positive(ctx: dict) -> str:
    return await groq_call(
        "Write a warm, genuine 2-sentence thank-you response to a customer "
        "who left very positive feedback about our product.",
        model=MODEL,
        temperature=0.7,
    )


async def draft_negative(ctx: dict) -> str:
    return await groq_call(
        "Write a professional, empathetic 2-sentence apology response to a customer "
        "who left negative feedback about our product.",
        model=MODEL,
        temperature=0.7,
    )


async def compose_reply(ctx: dict) -> str:
    pos  = ctx.get("draft_positive", "")
    neg  = ctx.get("draft_negative", "")
    body = pos or neg
    return await groq_call(
        f"Format this into a final customer service email reply. "
        f"Add a subject line and professional sign-off: '{body}'",
        model=MODEL,
        max_tokens=300,
        temperature=0.3,
    )


def build() -> ExecutionGraph:
    c_cfg = LLMConfig(model=MODEL, temperature=0.0, sim_latency_s=2.0)
    d_cfg = LLMConfig(model=MODEL, temperature=0.7, sim_latency_s=2.0)
    r_cfg = LLMConfig(model=MODEL, temperature=0.3, sim_latency_s=1.5)

    g = ExecutionGraph()
    g.add_node(Node(
        "classify_sentiment", NodeType.CONDITION, classify_sentiment,
        llm_config=c_cfg,
        true_branch="draft_positive", false_branch="draft_negative", p_true=0.8,
    ))
    g.add_node(Node(
        "draft_positive", NodeType.LLM_CALL, draft_positive,
        llm_config=d_cfg,
        metadata={"condition_branch": "classify_sentiment"},
    ))
    g.add_node(Node(
        "draft_negative", NodeType.LLM_CALL, draft_negative,
        llm_config=d_cfg,
        metadata={"condition_branch": "classify_sentiment"},
    ))
    g.add_node(Node(
        "compose_reply", NodeType.LLM_CALL, compose_reply,
        dependencies=["classify_sentiment"], llm_config=r_cfg,
    ))
    return g
