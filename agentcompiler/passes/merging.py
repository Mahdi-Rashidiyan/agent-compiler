"""
agentcompiler/passes/merging.py  —  Pass 2: LLM Call Merging

Chain merging: A→B→C (sequential, same config) → one multi-part prompt.
  Sequential cost:  N × (API_OH + gen) = N × sim_latency_s
  Merged cost:      API_OH + N × gen   < sequential

Parallel merging is SKIPPED for nodes already tagged by Pass 1 (parallel is
already better than merging for same-level nodes).
"""

from __future__ import annotations
import asyncio
from typing import Dict, List, Tuple

from agentcompiler.graph import ExecutionGraph, LLMConfig, Node, NodeType
from agentcompiler.passes.base import CompilerPass

API_OVERHEAD = 0.30   # seconds per API request (TLS + auth + routing)


class LLMCallMergingPass(CompilerPass):

    @property
    def name(self) -> str:
        return "LLMCallMerging"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        merge_count = 0

        # Strategy: chain merging only
        # (Parallel merging is skipped — parallel execution via Pass 1 is
        #  always faster than merging same-level calls.)
        chains = self._find_chains(graph)
        for chain in chains:
            if len(chain) >= 2:
                self._merge_chain(graph, chain)
                merge_count += 1

        print(f"  [Pass 2] Merging: {merge_count} chain group(s) merged")
        return graph

    # ─────────────────────────────────────────────────────────────────────────

    def _find_chains(self, graph: ExecutionGraph) -> List[List[str]]:
        """
        Find maximal chains A→B→C where:
          - every node is LLM_CALL with non-null llm_config
          - each node depends ONLY on its predecessor in the chain
          - predecessor has exactly one successor (no fan-out)
          - all share (model, temperature)
          - no node already absorbed or in a parallel group
        """
        succ: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
        for nid, node in graph.nodes.items():
            for dep in node.dependencies:
                if dep in succ:
                    succ[dep].append(nid)

        visited: set = set()
        chains: List[List[str]] = []

        for start in graph.nodes:
            if start in visited:
                continue
            node = graph.nodes[start]
            if not self._chain_eligible(node):
                continue

            chain = [start]
            current = start

            while True:
                successors = succ.get(current, [])
                if len(successors) != 1:
                    break
                nxt = successors[0]
                nxt_node = graph.nodes[nxt]
                if (not self._chain_eligible(nxt_node)
                        or nxt in visited
                        or len(nxt_node.dependencies) != 1
                        or not node.llm_config.mergeable_with(nxt_node.llm_config)):
                    break
                chain.append(nxt)
                current = nxt
                node = nxt_node

            if len(chain) >= 2:
                visited.update(chain)
                chains.append(chain)

        return chains

    def _chain_eligible(self, node: Node) -> bool:
        return (
            node.node_type == NodeType.LLM_CALL
            and node.llm_config is not None
            and "absorbed_by" not in node.metadata
            and "parallel_group" not in node.metadata   # don't touch Pass-1 groups
            and "condition_branch" not in node.metadata
        )

    def _merge_chain(self, graph: ExecutionGraph, nids: List[str]) -> None:
        first_node = graph.nodes[nids[0]]
        cfg        = first_node.llm_config
        n          = len(nids)

        gen_per_step   = max(cfg.sim_latency_s - API_OVERHEAD, 0.05)
        merged_latency = API_OVERHEAD + n * gen_per_step   # single API round-trip

        merged_cfg = LLMConfig(
            model=cfg.model,
            temperature=cfg.temperature,
            sim_latency_s=merged_latency,
        )

        merged_id = "merged_chain_" + "_".join(nids)

        async def merged_chain_fn(ctx: dict) -> dict:
            await asyncio.sleep(merged_cfg.sim_latency_s)
            results = {}
            for nid in nids:
                results[nid] = f"[chain-merged:{nid}]"
            return results

        merged_node = Node(
            id=merged_id,
            node_type=NodeType.LLM_CALL,
            fn=merged_chain_fn,
            dependencies=list(first_node.dependencies),
            llm_config=merged_cfg,
            metadata={"merged_from": nids, "is_merge": True, "merge_type": "chain"},
        )
        graph.add_node(merged_node)

        # Redirect last node's downstream to merged_id
        for other_id, other_node in graph.nodes.items():
            if other_id in nids or other_id == merged_id:
                continue
            if nids[-1] in other_node.dependencies:
                other_node.dependencies.remove(nids[-1])
                if merged_id not in other_node.dependencies:
                    other_node.dependencies.append(merged_id)

        for nid in nids:
            graph.nodes[nid].metadata["absorbed_by"] = merged_id
