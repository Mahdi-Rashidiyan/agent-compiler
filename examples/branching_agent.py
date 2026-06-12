"""
examples/branching_agent.py
============================
Agent 3 — Sentiment Branch Agent

classify_sentiment → (condition) → draft_positive  (p=0.8)
                                └─→ draft_negative  (p=0.2)
→ compose_reply

Branch nodes are tagged "condition_branch" so the scheduler ignores them —
they are run internally by the condition executor (either normally or
speculatively, depending on Pass 3).

Pass 3 win (P(positive) = 0.8 ≥ threshold 0.65):
  Without: classify (0.5s) → draft_positive (1.2s) → compose (0.3s) = 2.0s
  With:  [classify ∥ draft_positive] = max(0.5,1.2) = 1.2s → compose 0.3s = 1.5s
  Speedup: 1.33×
"""

import asyncio
from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType

CLASSIFY_LAT = 0.5
DRAFT_LAT    = 1.2
COMPOSE_LAT  = 0.3


async def classify_sentiment(ctx: dict) -> bool:
    await asyncio.sleep(CLASSIFY_LAT)
    return True   # deterministic for reproducible benchmarks


async def draft_positive(ctx: dict) -> str:
    await asyncio.sleep(DRAFT_LAT)
    return "Thank you so much! We're delighted you loved the experience."


async def draft_negative(ctx: dict) -> str:
    await asyncio.sleep(DRAFT_LAT)
    return "We sincerely apologise. Support will contact you within 24 hours."


async def compose_reply(ctx: dict) -> str:
    await asyncio.sleep(COMPOSE_LAT)
    pos = ctx.get("draft_positive", "")
    neg = ctx.get("draft_negative", "")
    return f"[REPLY] {pos or neg}"


def build() -> ExecutionGraph:
    c_cfg = LLMConfig(model="claude-3-haiku", temperature=0.0, sim_latency_s=CLASSIFY_LAT)
    d_cfg = LLMConfig(model="claude-3-haiku", temperature=0.7, sim_latency_s=DRAFT_LAT)
    r_cfg = LLMConfig(model="claude-3-haiku", temperature=0.3, sim_latency_s=COMPOSE_LAT)

    g = ExecutionGraph()
    g.add_node(Node(
        "classify_sentiment", NodeType.CONDITION, classify_sentiment,
        llm_config=c_cfg,
        true_branch="draft_positive", false_branch="draft_negative", p_true=0.8,
    ))
    # Branch nodes: tagged so the scheduler skips them (run by condition executor)
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
