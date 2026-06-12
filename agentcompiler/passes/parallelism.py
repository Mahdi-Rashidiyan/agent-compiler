"""
agentcompiler/passes/parallelism.py
=====================================
Pass 1 — Parallelism Extraction
================================

Finds groups of nodes at the same DAG level (i.e., no data dependency
between them) and tags them for concurrent execution.

This is the highest-impact pass.  Independent LLM calls that were
previously serialised are launched together with asyncio.gather(),
collapsing N × latency → 1 × latency on the critical path.

Example
-------
Before:  query_A (1s) → query_B (1s) → query_C (1s) → synthesize (0.8s)
         (queries are independent — sequential order was accidental)

After:   [query_A ∥ query_B ∥ query_C] (1s) → synthesize (0.8s)
         Total: 1.0 + 0.8 = 1.8s  vs  3.0 + 0.8 = 3.8s  →  2.1× speedup
"""

from __future__ import annotations

from typing import Dict, List

from agentcompiler.graph import ExecutionGraph, NodeType
from agentcompiler.passes.base import CompilerPass


class ParallelismExtractionPass(CompilerPass):

    @property
    def name(self) -> str:
        return "ParallelismExtraction"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        levels: Dict[str, int] = graph.compute_levels()
        max_lvl = max(levels.values(), default=0)

        parallel_groups: List[List[str]] = []

        for lvl in range(max_lvl + 1):
            # Candidate nodes: LLM calls or tools at the same level
            candidates = [
                nid
                for nid in graph.nodes_at_level(lvl, levels)
                if graph.nodes[nid].node_type in (NodeType.LLM_CALL, NodeType.TOOL)
            ]

            if len(candidates) > 1:
                group_key = f"pg_lvl{lvl}"
                for nid in candidates:
                    graph.nodes[nid].metadata["parallel_group"] = group_key
                parallel_groups.append(candidates)

        graph.metadata["parallel_groups"] = parallel_groups

        total_nodes = sum(len(g) for g in parallel_groups)
        print(
            f"  [Pass 1] Parallelism: {len(parallel_groups)} group(s), "
            f"{total_nodes} nodes will run concurrently"
        )
        return graph
