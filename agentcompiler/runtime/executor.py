"""
agentcompiler/runtime/executor.py
===================================
Async execution engine for compiled agent graphs.

Key semantic:
  - WITHOUT Pass 1 → nodes run ONE AT A TIME (truly sequential baseline).
    This mirrors a naive agent framework with no concurrency awareness.
  - WITH Pass 1   → nodes in the same parallel_group run with asyncio.gather().
    This is the compilation win: independent LLM calls overlap.
  - WITH Pass 2   → merged nodes replace N-call chains.
  - WITH Pass 3   → condition + predicted branch overlap.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Set

from agentcompiler.graph import ExecutionGraph, NodeType


class ExecutionContext:
    def __init__(self, input_data: Dict[str, Any]) -> None:
        self._data: Dict[str, Any] = {"__input__": input_data}
        self.completed: Set[str] = set()

    def set(self, node_id: str, value: Any) -> None:
        self._data[node_id] = value
        self.completed.add(node_id)

    def get(self, node_id: str) -> Any:
        return self._data.get(node_id)

    def dep_values(self, deps: List[str]) -> Dict[str, Any]:
        return {d: self._data.get(d) for d in deps}

    @property
    def data(self) -> Dict[str, Any]:
        return dict(self._data)


class AgentExecutor:

    async def run(
        self,
        graph: ExecutionGraph,
        input_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        ctx = ExecutionContext(input_data or {})

        # Nodes absorbed into a merge node — skip in scheduler
        absorbed: Set[str] = {
            nid for nid, n in graph.nodes.items()
            if "absorbed_by" in n.metadata
        }

        # Branch nodes are run INSIDE the condition executor, not by the scheduler
        branch_nodes: Set[str] = {
            nid for nid, n in graph.nodes.items()
            if "condition_branch" in n.metadata
        }

        skip = absorbed | branch_nodes

        while True:
            # Nodes ready to execute: all deps completed, not already done, not skipped
            ready = [
                nid for nid in graph.get_ready(ctx.completed | skip)
                if nid not in ctx.completed and nid not in skip
            ]
            if not ready:
                break

            # ── Group by parallel_group tag ──────────────────────────────────
            pg_map: Dict[str, List[str]] = {}
            solo: List[str] = []

            for nid in ready:
                pg = graph.nodes[nid].metadata.get("parallel_group")
                if pg:
                    pg_map.setdefault(pg, []).append(nid)
                else:
                    solo.append(nid)

            if pg_map:
                # ── Optimised path: run all parallel groups concurrently ──────
                # Solo nodes in the same scheduling batch also run concurrently
                # (they happen to be ready at the same time as a parallel group)
                coros, tags = [], []
                for pg_key, pg_nids in pg_map.items():
                    coros.append(self._run_parallel_group(graph, pg_nids, ctx))
                    tags.append(("group", pg_nids))
                for nid in solo:
                    coros.append(self._run_node(graph, nid, ctx))
                    tags.append(("single", nid))

                results = await asyncio.gather(*coros)
                for (kind, info), result in zip(tags, results):
                    if kind == "group":
                        for nid, res in zip(info, result):
                            ctx.set(nid, res)
                    else:
                        ctx.set(info, result)

            else:
                # ── Unoptimised / sequential path: ONE node at a time ────────
                # This is the correct baseline: a naive framework with no
                # concurrency awareness.  The compiler's job is to add pg tags
                # so this path is never taken for independent nodes.
                nid = solo[0]
                result = await self._run_node(graph, nid, ctx)
                ctx.set(nid, result)

        return ctx.data

    # ─────────────────────────────────────────────────────────────────────────

    async def _run_parallel_group(
        self,
        graph: ExecutionGraph,
        nids: List[str],
        ctx: ExecutionContext,
    ) -> List[Any]:
        return await asyncio.gather(
            *[self._run_node(graph, nid, ctx) for nid in nids]
        )

    async def _run_node(
        self,
        graph: ExecutionGraph,
        nid: str,
        ctx: ExecutionContext,
    ) -> Any:
        node = graph.nodes[nid]
        dep_vals = ctx.dep_values(node.dependencies)

        if node.metadata.get("is_merge"):
            return await self._run_merged(node, dep_vals, graph, ctx)

        if node.node_type == NodeType.CONDITION:
            return await self._run_condition(graph, node, dep_vals, ctx)

        return await node.fn(dep_vals)

    async def _run_merged(self, node, dep_vals, graph, ctx) -> Any:
        result = await node.fn(dep_vals)
        # Distribute sub-results back to absorbed siblings so downstream
        # nodes that look up their values find them in ctx
        if isinstance(result, dict):
            for orig_id, val in result.items():
                if orig_id in graph.nodes:
                    ctx.set(orig_id, val)
        return result

    async def _run_condition(self, graph, node, dep_vals, ctx) -> Any:
        speculate      = node.metadata.get("speculate")
        spec_target_id = node.metadata.get("speculative_target")

        if speculate and spec_target_id and spec_target_id in graph.nodes:
            # ── Speculative path ─────────────────────────────────────────────
            # Launch condition evaluation AND predicted branch simultaneously.
            spec_node     = graph.nodes[spec_target_id]
            spec_dep_vals = ctx.dep_values(spec_node.dependencies)

            condition_result, spec_result = await asyncio.gather(
                node.fn(dep_vals),
                spec_node.fn(spec_dep_vals),
            )

            correct_id = node.true_branch if condition_result else node.false_branch

            if correct_id == spec_target_id:
                # Prediction correct — pre-computed result is valid
                ctx.set(spec_target_id, spec_result)
            else:
                # Mis-speculation — discard and run the correct branch
                if correct_id and correct_id in graph.nodes:
                    correct_node = graph.nodes[correct_id]
                    correct_vals = ctx.dep_values(correct_node.dependencies)
                    correct_result = await correct_node.fn(correct_vals)
                    ctx.set(correct_id, correct_result)

            return condition_result

        else:
            # ── Non-speculative path ─────────────────────────────────────────
            # Evaluate condition first, then run the winning branch.
            condition_result = await node.fn(dep_vals)
            chosen_id = node.true_branch if condition_result else node.false_branch
            if chosen_id and chosen_id in graph.nodes:
                chosen_node = graph.nodes[chosen_id]
                chosen_vals = ctx.dep_values(chosen_node.dependencies)
                chosen_result = await chosen_node.fn(chosen_vals)
                ctx.set(chosen_id, chosen_result)
            return condition_result
