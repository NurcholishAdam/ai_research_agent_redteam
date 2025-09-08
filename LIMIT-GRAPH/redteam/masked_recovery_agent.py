# -*- coding: utf-8 -*-
"""
Masked Recovery Agent Module
Agent interface for masked graph recovery tasks
"""

import json
import time
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging

class MaskedRecoveryAgent(ABC):
    """
    Abstract base class for masked recovery agents
    """
    
    @abstractmethod
    def recover_masked_edges(self, query: str, masked_graph: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recover masked edges in graph
        
        Args:
            query: Query or task description
            masked_graph: Graph with masked edges
            context: Additional context for recovery
            
        Returns:
            Recovery results with predictions and reasoning
        """
        pass

class LimitGraphRecoveryAgent(MaskedRecoveryAgent):
    """
    LIMIT-GRAPH implementation of masked recovery agent
    """
    
    def __init__(self, graph_reasoner=None, entity_linker=None, memory_system=None):
        """Initialize with LIMIT-GRAPH components"""
        self.graph_reasoner = graph_reasoner
        self.entity_linker = entity_linker
        self.memory_system = memory_system
        self.logger = logging.getLogger(__name__)
        
    def recover_masked_edges(self, query: str, masked_graph: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recover masked edges using LIMIT-GRAPH reasoning
        """
        start_time = time.time()
        
        try:
            # Extract masked edges
            masked_edges = self._identify_masked_edges(masked_graph)
            
            # Initialize recovery results
            recovery_results = {
                "query": query,
                "masked_edges_count": len(masked_edges),
                "recovered_edges": [],
                "predictions": [],
                "reasoning": {},
                "trace": [],
                "confidence_scores": [],
                "processing_time": 0.0
            }
            
            # Process each masked edge
            for masked_edge in masked_edges:
                edge_recovery = self._recover_single_edge(
                    masked_edge, query, masked_graph, context
                )
                
                recovery_results["recovered_edges"].append(edge_recovery["recovered_edge"])
                recovery_results["predictions"].append(edge_recovery["prediction"])
                recovery_results["trace"].extend(edge_recovery["reasoning_steps"])
                recovery_results["confidence_scores"].append(edge_recovery["confidence"])
            
            # Generate overall reasoning
            recovery_results["reasoning"] = self._generate_overall_reasoning(
                recovery_results, query, context
            )
            
            recovery_results["processing_time"] = time.time() - start_time
            
            return recovery_results
            
        except Exception as e:
            self.logger.error(f"Error in masked edge recovery: {e}")
            return {
                "query": query,
                "error": str(e),
                "recovered_edges": [],
                "predictions": [],
                "processing_time": time.time() - start_time
            }
    
    def _identify_masked_edges(self, masked_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify edges with [MASK] relations"""
        masked_edges = []
        
        for edge in masked_graph.get("edges", []):
            if edge.get("relation") == "[MASK]":
                masked_edges.append(edge)
        
        return masked_edges
    
    def _recover_single_edge(self, masked_edge: Dict[str, Any], query: str,
                           masked_graph: Dict[str, Any], 
                           context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recover a single masked edge
        """
        source = masked_edge["source"]
        target = masked_edge["target"]
        
        # Initialize reasoning trace
        reasoning_steps = [
            f"Analyzing masked edge: {source} -> [MASK] -> {target}"
        ]
        
        # Step 1: Analyze graph context
        graph_context = self._analyze_graph_context(source, target, masked_graph)
        reasoning_steps.append(f"Graph context analysis: {graph_context['summary']}")
        
        # Step 2: Use entity linking if available
        entity_info = {}
        if self.entity_linker:
            try:
                entity_info = self._get_entity_information(source, target, query)
                reasoning_steps.append(f"Entity information retrieved: {len(entity_info)} entities")
            except Exception as e:
                reasoning_steps.append(f"Entity linking failed: {e}")
        
        # Step 3: Query memory system
        memory_context = {}
        if self.memory_system:
            try:
                memory_context = self._query_memory_system(source, target, query)
                reasoning_steps.append(f"Memory context: {memory_context.get('summary', 'No relevant memories')}")
            except Exception as e:
                reasoning_steps.append(f"Memory query failed: {e}")
        
        # Step 4: Reason about possible relations
        candidate_relations = self._generate_candidate_relations(
            source, target, graph_context, entity_info, memory_context, query
        )
        reasoning_steps.append(f"Generated {len(candidate_relations)} candidate relations")
        
        # Step 5: Select best relation
        best_relation, confidence = self._select_best_relation(
            candidate_relations, source, target, query, context
        )
        reasoning_steps.append(f"Selected relation: {best_relation} (confidence: {confidence:.2f})")
        
        # Create recovery result
        recovered_edge = {
            "source": source,
            "target": target,
            "relation": best_relation,
            "original_masked": True
        }
        
        prediction = {
            "source": source,
            "target": target,
            "predicted_relation": best_relation,
            "confidence": confidence,
            "alternatives": [rel["relation"] for rel in candidate_relations[:3]],
            "reasoning_summary": f"Predicted '{best_relation}' based on graph context and entity analysis"
        }
        
        return {
            "recovered_edge": recovered_edge,
            "prediction": prediction,
            "reasoning_steps": reasoning_steps,
            "confidence": confidence,
            "candidates": candidate_relations
        }
    
    def _analyze_graph_context(self, source: str, target: str, 
                             masked_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze graph structure around masked edge"""
        edges = masked_graph.get("edges", [])
        
        # Find edges involving source and target
        source_edges = [e for e in edges if e["source"] == source or e["target"] == source]
        target_edges = [e for e in edges if e["source"] == target or e["target"] == target]
        
        # Analyze relation patterns
        source_relations = [e["relation"] for e in source_edges if e["relation"] != "[MASK]"]
        target_relations = [e["relation"] for e in target_edges if e["relation"] != "[MASK]"]
        
        # Find common relation types
        all_relations = [e["relation"] for e in edges if e["relation"] != "[MASK]"]
        relation_counts = {}
        for rel in all_relations:
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
        
        return {
            "source_edges": len(source_edges),
            "target_edges": len(target_edges),
            "source_relations": source_relations,
            "target_relations": target_relations,
            "common_relations": sorted(relation_counts.items(), key=lambda x: x[1], reverse=True),
            "summary": f"Source has {len(source_edges)} edges, target has {len(target_edges)} edges"
        }
    
    def _get_entity_information(self, source: str, target: str, query: str) -> Dict[str, Any]:
        """Get entity information using entity linker"""
        if not self.entity_linker:
            return {}
        
        try:
            # Use entity linker to get information about entities
            source_info = self.entity_linker.get_entity_info(source)
            target_info = self.entity_linker.get_entity_info(target)
            
            return {
                "source_info": source_info,
                "target_info": target_info,
                "entity_types": {
                    source: source_info.get("type", "unknown"),
                    target: target_info.get("type", "unknown")
                }
            }
        except Exception as e:
            self.logger.warning(f"Entity linking failed: {e}")
            return {}
    
    def _query_memory_system(self, source: str, target: str, query: str) -> Dict[str, Any]:
        """Query memory system for relevant information"""
        if not self.memory_system:
            return {}
        
        try:
            # Query memory for relations between entities
            memory_query = f"relation between {source} and {target}"
            memories = self.memory_system.search(memory_query, limit=5)
            
            return {
                "relevant_memories": memories,
                "memory_count": len(memories),
                "summary": f"Found {len(memories)} relevant memories"
            }
        except Exception as e:
            self.logger.warning(f"Memory query failed: {e}")
            return {}
    
    def _generate_candidate_relations(self, source: str, target: str,
                                    graph_context: Dict[str, Any],
                                    entity_info: Dict[str, Any],
                                    memory_context: Dict[str, Any],
                                    query: str) -> List[Dict[str, Any]]:
        """Generate candidate relations for masked edge"""
        candidates = []
        
        # Strategy 1: Use common relations from graph
        common_relations = graph_context.get("common_relations", [])
        for relation, count in common_relations[:5]:
            candidates.append({
                "relation": relation,
                "score": count / 10.0,  # Normalize by frequency
                "source": "graph_frequency"
            })
        
        # Strategy 2: Use entity type patterns
        entity_types = entity_info.get("entity_types", {})
        source_type = entity_types.get(source, "unknown")
        target_type = entity_types.get(target, "unknown")
        
        # Common relation patterns by entity type
        type_patterns = {
            ("person", "person"): ["friend", "parent", "child", "spouse", "colleague"],
            ("person", "object"): ["owns", "has", "uses", "likes"],
            ("person", "location"): ["lives_in", "works_at", "visits"],
            ("object", "location"): ["located_in", "stored_at"],
            ("unknown", "unknown"): ["related_to", "associated_with"]
        }
        
        pattern_key = (source_type.lower(), target_type.lower())
        if pattern_key not in type_patterns:
            pattern_key = ("unknown", "unknown")
        
        for relation in type_patterns[pattern_key]:
            candidates.append({
                "relation": relation,
                "score": 0.6,
                "source": "entity_type_pattern"
            })
        
        # Strategy 3: Use memory-based relations
        memories = memory_context.get("relevant_memories", [])
        for memory in memories:
            if "relation" in memory:
                candidates.append({
                    "relation": memory["relation"],
                    "score": memory.get("confidence", 0.5),
                    "source": "memory"
                })
        
        # Strategy 4: Query-based inference
        query_lower = query.lower()
        query_relations = []
        
        if "who" in query_lower and ("own" in query_lower or "has" in query_lower):
            query_relations.extend(["owns", "has", "possesses"])
        elif "where" in query_lower:
            query_relations.extend(["located_in", "at", "in"])
        elif "what" in query_lower and "do" in query_lower:
            query_relations.extend(["does", "performs", "works_as"])
        
        for relation in query_relations:
            candidates.append({
                "relation": relation,
                "score": 0.7,
                "source": "query_inference"
            })
        
        # Remove duplicates and sort by score
        unique_candidates = {}
        for candidate in candidates:
            relation = candidate["relation"]
            if relation not in unique_candidates or candidate["score"] > unique_candidates[relation]["score"]:
                unique_candidates[relation] = candidate
        
        sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x["score"], reverse=True)
        
        return sorted_candidates[:10]  # Return top 10 candidates
    
    def _select_best_relation(self, candidates: List[Dict[str, Any]], 
                            source: str, target: str, query: str,
                            context: Optional[Dict[str, Any]]) -> tuple[str, float]:
        """Select the best relation from candidates"""
        if not candidates:
            return "related_to", 0.3  # Default fallback
        
        # Use the highest scoring candidate
        best_candidate = candidates[0]
        
        # Adjust confidence based on multiple factors
        base_confidence = best_candidate["score"]
        
        # Boost confidence if multiple strategies agree
        relation = best_candidate["relation"]
        agreement_count = sum(1 for c in candidates if c["relation"] == relation)
        if agreement_count > 1:
            base_confidence = min(base_confidence + 0.2, 1.0)
        
        # Boost confidence for common relations
        common_relations = ["owns", "has", "is", "located_in", "friend", "parent"]
        if relation in common_relations:
            base_confidence = min(base_confidence + 0.1, 1.0)
        
        return relation, base_confidence
    
    def _generate_overall_reasoning(self, recovery_results: Dict[str, Any],
                                 query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall reasoning summary"""
        recovered_count = len(recovery_results["recovered_edges"])
        avg_confidence = sum(recovery_results["confidence_scores"]) / max(len(recovery_results["confidence_scores"]), 1)
        
        reasoning = {
            "approach": "multi-strategy_masked_recovery",
            "strategies_used": [
                "graph_context_analysis",
                "entity_type_inference", 
                "memory_retrieval",
                "query_pattern_matching"
            ],
            "recovered_edges_count": recovered_count,
            "average_confidence": avg_confidence,
            "confidence": avg_confidence,
            "summary": f"Successfully recovered {recovered_count} masked edges with average confidence {avg_confidence:.2f}",
            "methodology": "Used graph structure analysis, entity type patterns, memory retrieval, and query inference to predict masked relations"
        }
        
        return reasoning

class SimpleRecoveryAgent(MaskedRecoveryAgent):
    """
    Simple baseline implementation for comparison
    """
    
    def __init__(self):
        """Initialize simple agent"""
        self.common_relations = [
            "owns", "has", "is", "located_in", "friend", "parent", "child",
            "works_at", "lives_in", "uses", "likes", "related_to"
        ]
    
    def recover_masked_edges(self, query: str, masked_graph: Dict[str, Any], 
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simple recovery using most common relations
        """
        start_time = time.time()
        
        # Find masked edges
        masked_edges = [e for e in masked_graph.get("edges", []) if e.get("relation") == "[MASK]"]
        
        # Simple strategy: use most common relation
        recovered_edges = []
        predictions = []
        
        for edge in masked_edges:
            # Use first common relation as prediction
            predicted_relation = self.common_relations[0]  # "owns"
            
            recovered_edges.append({
                "source": edge["source"],
                "target": edge["target"],
                "relation": predicted_relation
            })
            
            predictions.append({
                "source": edge["source"],
                "target": edge["target"],
                "predicted_relation": predicted_relation,
                "confidence": 0.5,
                "alternatives": self.common_relations[1:4]
            })
        
        return {
            "query": query,
            "recovered_edges": recovered_edges,
            "predictions": predictions,
            "reasoning": {
                "approach": "simple_baseline",
                "confidence": 0.5,
                "summary": f"Used most common relation '{self.common_relations[0]}' for all masked edges"
            },
            "trace": ["Applied simple baseline strategy using most common relations"],
            "processing_time": time.time() - start_time
        }