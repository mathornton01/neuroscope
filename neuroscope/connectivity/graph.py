"""
Functional connectivity graph construction and analysis.

Builds a graph where nodes are features/neurons/attention heads and
edges represent functional co-activation patterns. Enables discovery
of "default mode networks," hub neurons, and processing pathways.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class FunctionalNode:
    """A node in the functional connectivity graph."""
    node_id: str
    layer_idx: int
    node_type: str  # "feature", "neuron", "attention_head"
    mean_activation: float = 0.0
    activation_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class FunctionalEdge:
    """An edge representing co-activation between two nodes."""
    source_id: str
    target_id: str
    weight: float  # correlation strength
    co_activation_count: int = 0
    edge_type: str = "co_activation"  # or "causal", "information_flow"


@dataclass
class FunctionalNetwork:
    """A discovered functional network (group of co-activating nodes)."""
    network_id: str
    nodes: list[str]  # node IDs
    mean_internal_connectivity: float
    label: Optional[str] = None  # e.g., "syntax_network", "factual_recall"


class ConnectivityGraph:
    """
    Builds and analyzes functional connectivity graphs from activation data.

    Workflow:
    1. Collect activation snapshots across many inputs
    2. Compute pairwise correlations between features/neurons
    3. Threshold and build the graph
    4. Discover functional networks via community detection
    5. Identify hub nodes and processing pathways
    """

    def __init__(self, correlation_threshold: float = 0.3):
        self.threshold = correlation_threshold
        self.nodes: dict[str, FunctionalNode] = {}
        self.edges: list[FunctionalEdge] = []
        self._activation_history: list[dict[str, float]] = []

    def record_activations(self, activations: dict[str, float]):
        """
        Record a single activation snapshot.

        Args:
            activations: Dict mapping node_id -> activation_value
                        for a single input.
        """
        self._activation_history.append(activations)

        # Update node stats
        for node_id, value in activations.items():
            if node_id not in self.nodes:
                # Parse node_id format: "layer_{idx}_{type}_{feature_idx}"
                parts = node_id.split("_")
                layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1
                node_type = parts[2] if len(parts) > 2 else "unknown"

                self.nodes[node_id] = FunctionalNode(
                    node_id=node_id,
                    layer_idx=layer_idx,
                    node_type=node_type,
                )

            node = self.nodes[node_id]
            n = node.activation_count
            node.mean_activation = (node.mean_activation * n + value) / (n + 1)
            node.activation_count += 1

    def build_graph(self, method: str = "pearson") -> "ConnectivityGraph":
        """
        Compute functional connectivity from recorded activations.

        Args:
            method: Correlation method ("pearson", "spearman", "mutual_info")
        """
        if len(self._activation_history) < 10:
            raise ValueError(
                f"Need at least 10 activation snapshots, got {len(self._activation_history)}. "
                "Record more activations before building the graph."
            )

        # Build activation matrix
        all_nodes = sorted(self.nodes.keys())
        n_samples = len(self._activation_history)
        n_nodes = len(all_nodes)

        matrix = np.zeros((n_samples, n_nodes))
        for i, snapshot in enumerate(self._activation_history):
            for j, node_id in enumerate(all_nodes):
                matrix[i, j] = snapshot.get(node_id, 0.0)

        # Compute pairwise correlations
        if method == "pearson":
            corr_matrix = np.corrcoef(matrix.T)
        else:
            raise NotImplementedError(f"Method {method} not yet implemented")

        # Build edges from significant correlations
        self.edges = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                corr = corr_matrix[i, j]
                if not np.isnan(corr) and abs(corr) > self.threshold:
                    self.edges.append(
                        FunctionalEdge(
                            source_id=all_nodes[i],
                            target_id=all_nodes[j],
                            weight=float(corr),
                            co_activation_count=n_samples,
                        )
                    )

        return self

    def find_networks(self, min_size: int = 3) -> list[FunctionalNetwork]:
        """
        Discover functional networks using community detection.

        Simple greedy modularity-based approach. For production use,
        consider Louvain or spectral clustering.
        """
        # Build adjacency list
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            adjacency[edge.source_id].add(edge.target_id)
            adjacency[edge.target_id].add(edge.source_id)

        # Simple connected components as initial networks
        visited = set()
        networks = []

        for node_id in self.nodes:
            if node_id in visited:
                continue

            # BFS
            component = []
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                queue.extend(adjacency[current] - visited)

            if len(component) >= min_size:
                # Compute internal connectivity
                internal_edges = sum(
                    1 for e in self.edges
                    if e.source_id in component and e.target_id in component
                )
                max_edges = len(component) * (len(component) - 1) / 2
                density = internal_edges / max_edges if max_edges > 0 else 0

                networks.append(
                    FunctionalNetwork(
                        network_id=f"network_{len(networks)}",
                        nodes=component,
                        mean_internal_connectivity=density,
                    )
                )

        return sorted(networks, key=lambda n: len(n.nodes), reverse=True)

    def find_hubs(self, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Find hub nodes with highest degree centrality.

        Hub neurons are disproportionately connected and often
        critical for model function (analogous to connector hubs
        in neuroscience).
        """
        degree: dict[str, float] = defaultdict(float)
        for edge in self.edges:
            degree[edge.source_id] += abs(edge.weight)
            degree[edge.target_id] += abs(edge.weight)

        ranked = sorted(degree.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def get_stats(self) -> dict:
        """Summary statistics for the connectivity graph."""
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_snapshots": len(self._activation_history),
            "mean_edge_weight": (
                np.mean([e.weight for e in self.edges]) if self.edges else 0.0
            ),
            "graph_density": (
                2 * len(self.edges) / (len(self.nodes) * (len(self.nodes) - 1))
                if len(self.nodes) > 1
                else 0.0
            ),
        }
