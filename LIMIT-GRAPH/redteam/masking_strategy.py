# -*- coding: utf-8 -*-
"""
Masking Strategy Module
Inspired by SafeProtein's structural masking for graph adversarial evaluation
"""

import random
import json
from typing import Dict, List, Any, Tuple
from enum import Enum
import networkx as nx
import numpy as np

class MaskingType(Enum):
    """Types of masking strategies"""
    RANDOM = "random"
    STRUCTURAL = "structural" 
    CRITICAL_PATH = "critical_path"
    ADVERSARIAL = "adversarial"
    SEMANTIC = "semantic"

class MaskingStrategy:
    """
    Implements various masking strategies for graph red-teaming
    """
    
    def __init__(self, seed: int = 42):
        """Initialize masking strategy with random seed"""
        random.seed(seed)
        np.random.seed(seed)
        
    def apply_masking(self, graph: Dict[str, Any], strategy: MaskingType, 
                     mask_ratio: float = 0.3) -> Dict[str, Any]:
        """
        Apply masking strategy to graph
        
        Args:
            graph: Original graph structure
            strategy: Type of masking to apply
            mask_ratio: Proportion of edges/nodes to mask
            
        Returns:
            Masked graph with ground truth for evaluation
        """
        if strategy == MaskingType.RANDOM:
            return self._random_masking(graph, mask_ratio)
        elif strategy == MaskingType.STRUCTURAL:
            return self._structural_masking(graph, mask_ratio)
        elif strategy == MaskingType.CRITICAL_PATH:
            return self._critical_path_masking(graph, mask_ratio)
        elif strategy == MaskingType.ADVERSARIAL:
            return self._adversarial_masking(graph, mask_ratio)
        elif strategy == MaskingType.SEMANTIC:
            return self._semantic_masking(graph, mask_ratio)
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
    
    def _random_masking(self, graph: Dict[str, Any], mask_ratio: float) -> Dict[str, Any]:
        """Random edge masking"""
        edges = graph.get("edges", [])
        num_to_mask = int(len(edges) * mask_ratio)
        
        masked_indices = random.sample(range(len(edges)), num_to_mask)
        masked_edges = []
        ground_truth = []
        
        for i, edge in enumerate(edges):
            if i in masked_indices:
                masked_edge = edge.copy()
                ground_truth.append({
                    "original_relation": edge["relation"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "mask_type": "random"
                })
                masked_edge["relation"] = "[MASK]"
                masked_edges.append(masked_edge)
            else:
                masked_edges.append(edge)
        
        return {
            "masked_graph": {"edges": masked_edges, "nodes": graph.get("nodes", [])},
            "ground_truth": ground_truth,
            "masking_strategy": "random",
            "mask_ratio": mask_ratio
        }
    
    def _structural_masking(self, graph: Dict[str, Any], mask_ratio: float) -> Dict[str, Any]:
        """Structural masking targeting high-degree nodes"""
        edges = graph.get("edges", [])
        
        # Build NetworkX graph for analysis
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        
        # Calculate node degrees and centrality
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        # Score edges by structural importance
        edge_scores = []
        for i, edge in enumerate(edges):
            source_importance = degree_centrality.get(edge["source"], 0) + \
                              betweenness_centrality.get(edge["source"], 0)
            target_importance = degree_centrality.get(edge["target"], 0) + \
                              betweenness_centrality.get(edge["target"], 0)
            
            edge_scores.append((i, source_importance + target_importance))
        
        # Sort by importance and mask top edges
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        num_to_mask = int(len(edges) * mask_ratio)
        masked_indices = [score[0] for score in edge_scores[:num_to_mask]]
        
        masked_edges = []
        ground_truth = []
        
        for i, edge in enumerate(edges):
            if i in masked_indices:
                masked_edge = edge.copy()
                ground_truth.append({
                    "original_relation": edge["relation"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "mask_type": "structural",
                    "importance_score": edge_scores[masked_indices.index(i)][1]
                })
                masked_edge["relation"] = "[MASK]"
                masked_edges.append(masked_edge)
            else:
                masked_edges.append(edge)
        
        return {
            "masked_graph": {"edges": masked_edges, "nodes": graph.get("nodes", [])},
            "ground_truth": ground_truth,
            "masking_strategy": "structural",
            "mask_ratio": mask_ratio
        }
    
    def _critical_path_masking(self, graph: Dict[str, Any], mask_ratio: float) -> Dict[str, Any]:
        """Mask edges on critical paths between important nodes"""
        edges = graph.get("edges", [])
        
        # Build NetworkX graph
        G = nx.DiGraph()
        for edge in edges:
            G.add_edge(edge["source"], edge["target"], relation=edge["relation"])
        
        # Find critical paths
        critical_edges = set()
        nodes = list(G.nodes())
        
        # Sample node pairs and find shortest paths
        for _ in range(min(50, len(nodes) * len(nodes))):
            source = random.choice(nodes)
            target = random.choice(nodes)
            
            if source != target and nx.has_path(G, source, target):
                try:
                    path = nx.shortest_path(G, source, target)
                    for i in range(len(path) - 1):
                        critical_edges.add((path[i], path[i + 1]))
                except nx.NetworkXNoPath:
                    continue
        
        # Mask edges on critical paths
        masked_edges = []
        ground_truth = []
        masked_count = 0
        target_mask_count = int(len(edges) * mask_ratio)
        
        for edge in edges:
            edge_tuple = (edge["source"], edge["target"])
            
            if (edge_tuple in critical_edges and 
                masked_count < target_mask_count):
                
                masked_edge = edge.copy()
                ground_truth.append({
                    "original_relation": edge["relation"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "mask_type": "critical_path"
                })
                masked_edge["relation"] = "[MASK]"
                masked_edges.append(masked_edge)
                masked_count += 1
            else:
                masked_edges.append(edge)
        
        return {
            "masked_graph": {"edges": masked_edges, "nodes": graph.get("nodes", [])},
            "ground_truth": ground_truth,
            "masking_strategy": "critical_path",
            "mask_ratio": mask_ratio
        }
    
    def _adversarial_masking(self, graph: Dict[str, Any], mask_ratio: float) -> Dict[str, Any]:
        """Adversarial masking to create ambiguous scenarios"""
        edges = graph.get("edges", [])
        
        # Group edges by relation type
        relation_groups = {}
        for i, edge in enumerate(edges):
            relation = edge["relation"]
            if relation not in relation_groups:
                relation_groups[relation] = []
            relation_groups[relation].append((i, edge))
        
        # Target relations with multiple instances for ambiguity
        ambiguous_relations = {rel: edges_list for rel, edges_list in relation_groups.items() 
                             if len(edges_list) > 1}
        
        masked_edges = []
        ground_truth = []
        masked_count = 0
        target_mask_count = int(len(edges) * mask_ratio)
        
        # Prioritize masking edges with ambiguous relations
        for relation, edge_list in ambiguous_relations.items():
            if masked_count >= target_mask_count:
                break
                
            # Mask some edges of this relation type
            num_to_mask = min(len(edge_list) // 2, target_mask_count - masked_count)
            masked_in_relation = random.sample(edge_list, num_to_mask)
            
            for idx, edge in masked_in_relation:
                ground_truth.append({
                    "original_relation": edge["relation"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "mask_type": "adversarial",
                    "ambiguity_level": len(edge_list)
                })
                masked_count += 1
        
        # Create masked edges list
        masked_indices = {gt["source"] + "->" + gt["target"] for gt in ground_truth}
        
        for edge in edges:
            edge_key = edge["source"] + "->" + edge["target"]
            if edge_key in masked_indices:
                masked_edge = edge.copy()
                masked_edge["relation"] = "[MASK]"
                masked_edges.append(masked_edge)
            else:
                masked_edges.append(edge)
        
        return {
            "masked_graph": {"edges": masked_edges, "nodes": graph.get("nodes", [])},
            "ground_truth": ground_truth,
            "masking_strategy": "adversarial",
            "mask_ratio": mask_ratio
        }
    
    def _semantic_masking(self, graph: Dict[str, Any], mask_ratio: float) -> Dict[str, Any]:
        """Semantic masking based on relation similarity"""
        edges = graph.get("edges", [])
        
        # Simple semantic grouping by relation keywords
        semantic_groups = {
            "ownership": ["owns", "has", "possesses", "memiliki"],
            "location": ["in", "at", "located", "di"],
            "relationship": ["friend", "parent", "child", "teman"],
            "action": ["does", "performs", "melakukan"]
        }
        
        # Classify edges by semantic type
        edge_semantics = []
        for i, edge in enumerate(edges):
            relation = edge["relation"].lower()
            semantic_type = "other"
            
            for sem_type, keywords in semantic_groups.items():
                if any(keyword in relation for keyword in keywords):
                    semantic_type = sem_type
                    break
            
            edge_semantics.append((i, edge, semantic_type))
        
        # Mask edges from each semantic group
        masked_edges = []
        ground_truth = []
        target_mask_count = int(len(edges) * mask_ratio)
        
        # Group by semantic type
        semantic_buckets = {}
        for i, edge, sem_type in edge_semantics:
            if sem_type not in semantic_buckets:
                semantic_buckets[sem_type] = []
            semantic_buckets[sem_type].append((i, edge))
        
        # Distribute masking across semantic types
        masked_count = 0
        for sem_type, edge_list in semantic_buckets.items():
            if masked_count >= target_mask_count:
                break
                
            num_to_mask = min(len(edge_list) // 3, 
                            max(1, (target_mask_count - masked_count) // len(semantic_buckets)))
            
            if edge_list and num_to_mask > 0:
                masked_in_type = random.sample(edge_list, min(num_to_mask, len(edge_list)))
                
                for idx, edge in masked_in_type:
                    ground_truth.append({
                        "original_relation": edge["relation"],
                        "source": edge["source"],
                        "target": edge["target"],
                        "mask_type": "semantic",
                        "semantic_group": sem_type
                    })
                    masked_count += 1
        
        # Create final masked edges
        masked_indices = {gt["source"] + "->" + gt["target"] for gt in ground_truth}
        
        for edge in edges:
            edge_key = edge["source"] + "->" + edge["target"]
            if edge_key in masked_indices:
                masked_edge = edge.copy()
                masked_edge["relation"] = "[MASK]"
                masked_edges.append(masked_edge)
            else:
                masked_edges.append(edge)
        
        return {
            "masked_graph": {"edges": masked_edges, "nodes": graph.get("nodes", [])},
            "ground_truth": ground_truth,
            "masking_strategy": "semantic",
            "mask_ratio": mask_ratio
        }
    
    def generate_test_scenarios(self, base_graphs: List[Dict], 
                              strategies: List[MaskingType] = None,
                              mask_ratios: List[float] = None) -> List[Dict]:
        """
        Generate comprehensive test scenarios
        
        Args:
            base_graphs: List of base graph structures
            strategies: Masking strategies to apply
            mask_ratios: Different masking ratios to test
            
        Returns:
            List of masked graph scenarios for evaluation
        """
        if strategies is None:
            strategies = list(MaskingType)
        
        if mask_ratios is None:
            mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        scenarios = []
        
        for graph in base_graphs:
            for strategy in strategies:
                for ratio in mask_ratios:
                    try:
                        masked_result = self.apply_masking(graph, strategy, ratio)
                        
                        scenario = {
                            "scenario_id": f"{strategy.value}_{ratio}_{len(scenarios)}",
                            "original_graph": graph,
                            "masked_graph": masked_result["masked_graph"],
                            "ground_truth": masked_result["ground_truth"],
                            "masking_strategy": strategy.value,
                            "mask_ratio": ratio,
                            "difficulty_level": self._calculate_difficulty(masked_result)
                        }
                        
                        scenarios.append(scenario)
                        
                    except Exception as e:
                        print(f"Error generating scenario for {strategy.value} with ratio {ratio}: {e}")
                        continue
        
        return scenarios
    
    def _calculate_difficulty(self, masked_result: Dict) -> str:
        """Calculate difficulty level of masked scenario"""
        mask_ratio = masked_result["mask_ratio"]
        strategy = masked_result["masking_strategy"]
        
        # Base difficulty on mask ratio
        if mask_ratio <= 0.2:
            base_difficulty = "easy"
        elif mask_ratio <= 0.4:
            base_difficulty = "medium"
        else:
            base_difficulty = "hard"
        
        # Adjust based on strategy complexity
        strategy_multipliers = {
            "random": 1.0,
            "structural": 1.2,
            "critical_path": 1.4,
            "adversarial": 1.6,
            "semantic": 1.3
        }
        
        multiplier = strategy_multipliers.get(strategy, 1.0)
        
        if multiplier >= 1.5:
            if base_difficulty == "easy":
                return "medium"
            elif base_difficulty == "medium":
                return "hard"
            else:
                return "expert"
        elif multiplier >= 1.3:
            if base_difficulty == "hard":
                return "expert"
        
        return base_difficulty