"""
agentcompiler/passes/speculative.py
=====================================
Pass 3 — Speculative Branch Execution
=======================================

For CONDITION nodes where one branch has a high prior probability, the
compiler pre-starts that branch's execution *concurrently* with the
condition evaluation.

Timeline without speculation:
  ──[condition: 0.5s]──[branch: 1.2s]──[output: 0.3s]──   total: 2.0s

Timeline with speculation (P(true) = 0.8):
  ──[condition: 0.5s]──────────────────[output: 0.3s]──   total: 1.5s
     ──[true_branch: 1.2s]────────────────┘

  → branch finishes BEFORE condition resolves → eliminated from critical path.

If the prediction is wrong (20% of the time): the speculative result is
discarded and the correct branch runs normally (pessimistic case: 2.0 + 1.2s).

Expected latency = P_correct × T_hit + P_wrong × T_miss

The threshold (default 0.65) ensures we only speculate when the probability
gain exceeds the mis-speculation cost.
"""

from agentcompiler.graph import ExecutionGraph, NodeType
from agentcompiler.passes.base import CompilerPass


class SpeculativeBranchPass(CompilerPass):

    THRESHOLD = 0.65   # minimum confidence to trigger speculation

    @property
    def name(self) -> str:
        return "SpeculativeBranchExecution"

    def apply(self, graph: ExecutionGraph) -> ExecutionGraph:
        spec_count = 0

        for nid, node in graph.nodes.items():
            if node.node_type != NodeType.CONDITION:
                continue

            # Choose the branch with higher prior
            if node.p_true >= self.THRESHOLD and node.true_branch:
                node.metadata["speculate"]         = "true_branch"
                node.metadata["speculative_target"] = node.true_branch
                spec_count += 1

            elif (1.0 - node.p_true) >= self.THRESHOLD and node.false_branch:
                node.metadata["speculate"]         = "false_branch"
                node.metadata["speculative_target"] = node.false_branch
                spec_count += 1

        print(
            f"  [Pass 3] Speculation: {spec_count} condition node(s) "
            f"flagged for speculative branch pre-execution"
        )
        return graph
