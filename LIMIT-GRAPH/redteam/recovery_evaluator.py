# -*- coding: utf-8 -*-
"""
Recovery Evaluator Module
Evaluates agent performance on masked graph recovery tasks
"""

import json
import time
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import logging

@dataclass
class RecoveryMetrics:
    """Metrics for recovery evaluation"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_score: float
    reasoning_quality: float
    trace_fidelity: float
    recovery_time: float

class RecoveryEvaluator:
    """
    Evaluates masked graph recovery performance
    """
    
    def __init__(self, logger=None):
        """Initialize evaluator"""
        self.logger = logger or logging.getLogger(__name__)
        self.evaluation_history = []
        
    def evaluate_recovery(self, agent_response: Dict[str, Any], 
                         ground_truth: List[Dict[str, Any]],
                         scenario_metadata: Dict[str, Any]) -> RecoveryMetrics:
        """
        Evaluate agent's recovery performance
        
        Args:
            agent_response: Agent's recovery attempt
            ground_truth: Expected recovery results
            scenario_metadata: Scenario information
            
        Returns:
            RecoveryMetrics with detailed evaluation
        """
        start_time = time.time()
        
        # Extract recovered relations
        recovered_relations = self._extract_recovered_relations(agent_response)
        expected_relations = self._extract_expected_relations(ground_truth)
        
        # Calculate core metrics
        accuracy = self._calculate_accuracy(recovered_relations, expected_relations)
        precision = self._calculate_precision(recovered_relations, expected_relations)
        recall = self._calculate_recall(recovered_relations, expected_relations)
        f1_score = self._calculate_f1(precision, recall)
        
        # Calculate advanced metrics
        confidence_score = self._evaluate_confidence(agent_response)
        reasoning_quality = self._evaluate_reasoning_quality(agent_response, ground_truth)
        trace_fidelity = self._evaluate_trace_fidelity(agent_response, scenario_metadata)
        
        recovery_time = time.time() - start_time
        
        metrics = RecoveryMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confidence_score=confidence_score,
            reasoning_quality=reasoning_quality,
            trace_fidelity=trace_fidelity,
            recovery_time=recovery_time
        )
        
        # Store evaluation result
        evaluation_result = {
            "timestamp": time.time(),
            "scenario_id": scenario_metadata.get("scenario_id", "unknown"),
            "masking_strategy": scenario_metadata.get("masking_strategy", "unknown"),
            "mask_ratio": scenario_metadata.get("mask_ratio", 0.0),
            "metrics": metrics,
            "agent_response": agent_response,
            "ground_truth": ground_truth
        }
        
        self.evaluation_history.append(evaluation_result)
        
        return metrics
    
    def _extract_recovered_relations(self, agent_response: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract recovered relations from agent response"""
        recovered = []
        
        if "recovered_edges" in agent_response:
            for edge in agent_response["recovered_edges"]:
                recovered.append({
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "relation": edge.get("relation", "")
                })
        
        if "predictions" in agent_response:
            for pred in agent_response["predictions"]:
                recovered.append({
                    "source": pred.get("source", ""),
                    "target": pred.get("target", ""),
                    "relation": pred.get("predicted_relation", "")
                })
        
        return recovered
    
    def _extract_expected_relations(self, ground_truth: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract expected relations from ground truth"""
        expected = []
        
        for gt in ground_truth:
            expected.append({
                "source": gt.get("source", ""),
                "target": gt.get("target", ""),
                "relation": gt.get("original_relation", "")
            })
        
        return expected
    
    def _calculate_accuracy(self, recovered: List[Dict], expected: List[Dict]) -> float:
        """Calculate exact match accuracy"""
        if not expected:
            return 1.0 if not recovered else 0.0
        
        correct = 0
        for exp in expected:
            for rec in recovered:
                if (exp["source"] == rec["source"] and 
                    exp["target"] == rec["target"] and
                    exp["relation"].lower() == rec["relation"].lower()):
                    correct += 1
                    break
        
        return correct / len(expected)
    
    def _calculate_precision(self, recovered: List[Dict], expected: List[Dict]) -> float:
        """Calculate precision (correct recoveries / total recoveries)"""
        if not recovered:
            return 1.0 if not expected else 0.0
        
        correct = 0
        for rec in recovered:
            for exp in expected:
                if (exp["source"] == rec["source"] and 
                    exp["target"] == rec["target"] and
                    exp["relation"].lower() == rec["relation"].lower()):
                    correct += 1
                    break
        
        return correct / len(recovered)
    
    def _calculate_recall(self, recovered: List[Dict], expected: List[Dict]) -> float:
        """Calculate recall (correct recoveries / total expected)"""
        if not expected:
            return 1.0
        
        correct = 0
        for exp in expected:
            for rec in recovered:
                if (exp["source"] == rec["source"] and 
                    exp["target"] == rec["target"] and
                    exp["relation"].lower() == rec["relation"].lower()):
                    correct += 1
                    break
        
        return correct / len(expected)
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _evaluate_confidence(self, agent_response: Dict[str, Any]) -> float:
        """Evaluate confidence scores in agent response"""
        confidence_scores = []
        
        # Extract confidence from predictions
        if "predictions" in agent_response:
            for pred in agent_response["predictions"]:
                if "confidence" in pred:
                    confidence_scores.append(pred["confidence"])
        
        # Extract confidence from reasoning
        if "reasoning" in agent_response:
            reasoning = agent_response["reasoning"]
            if isinstance(reasoning, dict) and "confidence" in reasoning:
                confidence_scores.append(reasoning["confidence"])
        
        # Default confidence evaluation based on response completeness
        if not confidence_scores:
            completeness = len(agent_response.get("recovered_edges", [])) > 0
            confidence_scores.append(0.7 if completeness else 0.3)
        
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _evaluate_reasoning_quality(self, agent_response: Dict[str, Any], 
                                  ground_truth: List[Dict[str, Any]]) -> float:
        """Evaluate quality of reasoning process"""
        reasoning_score = 0.0
        
        # Check if reasoning is provided
        if "reasoning" in agent_response:
            reasoning = agent_response["reasoning"]
            
            # Score based on reasoning structure
            if isinstance(reasoning, dict):
                if "steps" in reasoning:
                    reasoning_score += 0.3  # Has structured steps
                if "evidence" in reasoning:
                    reasoning_score += 0.2  # Provides evidence
                if "context_used" in reasoning:
                    reasoning_score += 0.2  # Uses context
            elif isinstance(reasoning, str) and len(reasoning) > 50:
                reasoning_score += 0.4  # Has substantial reasoning text
        
        # Check if agent used graph traversal
        if "traversal_path" in agent_response:
            reasoning_score += 0.2
        
        # Check if agent considered multiple hypotheses
        if "alternative_predictions" in agent_response:
            reasoning_score += 0.1
        
        # Bonus for mentioning uncertainty
        reasoning_text = str(agent_response.get("reasoning", "")).lower()
        uncertainty_keywords = ["uncertain", "possible", "might", "could", "maybe"]
        if any(keyword in reasoning_text for keyword in uncertainty_keywords):
            reasoning_score += 0.1
        
        return min(reasoning_score, 1.0)
    
    def _evaluate_trace_fidelity(self, agent_response: Dict[str, Any], 
                               scenario_metadata: Dict[str, Any]) -> float:
        """Evaluate fidelity of reasoning trace"""
        trace_score = 0.0
        
        # Check if trace is provided
        if "trace" in agent_response or "reasoning_trace" in agent_response:
            trace_score += 0.4
        
        # Check if trace mentions masking strategy awareness
        trace_text = str(agent_response.get("trace", "") + 
                        agent_response.get("reasoning_trace", "")).lower()
        
        strategy = scenario_metadata.get("masking_strategy", "")
        if strategy in trace_text:
            trace_score += 0.2
        
        # Check for graph structure awareness
        graph_keywords = ["graph", "edge", "node", "connection", "relation"]
        if any(keyword in trace_text for keyword in graph_keywords):
            trace_score += 0.2
        
        # Check for context utilization
        context_keywords = ["context", "memory", "previous", "prior"]
        if any(keyword in trace_text for keyword in context_keywords):
            trace_score += 0.2
        
        return min(trace_score, 1.0)
    
    def generate_evaluation_report(self, scenario_filter: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        
        Args:
            scenario_filter: Filter criteria for scenarios to include
            
        Returns:
            Detailed evaluation report
        """
        filtered_evaluations = self._filter_evaluations(scenario_filter)
        
        if not filtered_evaluations:
            return {"error": "No evaluations match the filter criteria"}
        
        # Aggregate metrics
        metrics_by_strategy = defaultdict(list)
        metrics_by_difficulty = defaultdict(list)
        overall_metrics = []
        
        for eval_result in filtered_evaluations:
            strategy = eval_result["masking_strategy"]
            metrics = eval_result["metrics"]
            
            metrics_by_strategy[strategy].append(metrics)
            overall_metrics.append(metrics)
            
            # Determine difficulty level
            mask_ratio = eval_result["mask_ratio"]
            if mask_ratio <= 0.2:
                difficulty = "easy"
            elif mask_ratio <= 0.4:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            metrics_by_difficulty[difficulty].append(metrics)
        
        # Calculate aggregate statistics
        report = {
            "evaluation_summary": {
                "total_scenarios": len(filtered_evaluations),
                "strategies_tested": list(metrics_by_strategy.keys()),
                "evaluation_period": {
                    "start": min(e["timestamp"] for e in filtered_evaluations),
                    "end": max(e["timestamp"] for e in filtered_evaluations)
                }
            },
            "overall_performance": self._calculate_aggregate_metrics(overall_metrics),
            "performance_by_strategy": {
                strategy: self._calculate_aggregate_metrics(metrics_list)
                for strategy, metrics_list in metrics_by_strategy.items()
            },
            "performance_by_difficulty": {
                difficulty: self._calculate_aggregate_metrics(metrics_list)
                for difficulty, metrics_list in metrics_by_difficulty.items()
            },
            "detailed_results": filtered_evaluations
        }
        
        return report
    
    def _filter_evaluations(self, filter_criteria: Dict[str, Any] = None) -> List[Dict]:
        """Filter evaluation history based on criteria"""
        if not filter_criteria:
            return self.evaluation_history
        
        filtered = []
        for eval_result in self.evaluation_history:
            include = True
            
            for key, value in filter_criteria.items():
                if key in eval_result and eval_result[key] != value:
                    include = False
                    break
            
            if include:
                filtered.append(eval_result)
        
        return filtered
    
    def _calculate_aggregate_metrics(self, metrics_list: List[RecoveryMetrics]) -> Dict[str, float]:
        """Calculate aggregate statistics for metrics"""
        if not metrics_list:
            return {}
        
        return {
            "accuracy": {
                "mean": np.mean([m.accuracy for m in metrics_list]),
                "std": np.std([m.accuracy for m in metrics_list]),
                "min": np.min([m.accuracy for m in metrics_list]),
                "max": np.max([m.accuracy for m in metrics_list])
            },
            "precision": {
                "mean": np.mean([m.precision for m in metrics_list]),
                "std": np.std([m.precision for m in metrics_list])
            },
            "recall": {
                "mean": np.mean([m.recall for m in metrics_list]),
                "std": np.std([m.recall for m in metrics_list])
            },
            "f1_score": {
                "mean": np.mean([m.f1_score for m in metrics_list]),
                "std": np.std([m.f1_score for m in metrics_list])
            },
            "confidence_score": {
                "mean": np.mean([m.confidence_score for m in metrics_list]),
                "std": np.std([m.confidence_score for m in metrics_list])
            },
            "reasoning_quality": {
                "mean": np.mean([m.reasoning_quality for m in metrics_list]),
                "std": np.std([m.reasoning_quality for m in metrics_list])
            },
            "trace_fidelity": {
                "mean": np.mean([m.trace_fidelity for m in metrics_list]),
                "std": np.std([m.trace_fidelity for m in metrics_list])
            },
            "recovery_time": {
                "mean": np.mean([m.recovery_time for m in metrics_list]),
                "std": np.std([m.recovery_time for m in metrics_list])
            }
        }
    
    def export_results(self, filepath: str, format_type: str = "json") -> bool:
        """
        Export evaluation results to file
        
        Args:
            filepath: Output file path
            format_type: Export format (json, csv)
            
        Returns:
            Success status
        """
        try:
            if format_type.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.evaluation_history, f, indent=2, default=str)
            
            elif format_type.lower() == "csv":
                import pandas as pd
                
                # Flatten evaluation data for CSV
                flattened_data = []
                for eval_result in self.evaluation_history:
                    metrics = eval_result["metrics"]
                    row = {
                        "timestamp": eval_result["timestamp"],
                        "scenario_id": eval_result["scenario_id"],
                        "masking_strategy": eval_result["masking_strategy"],
                        "mask_ratio": eval_result["mask_ratio"],
                        "accuracy": metrics.accuracy,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "confidence_score": metrics.confidence_score,
                        "reasoning_quality": metrics.reasoning_quality,
                        "trace_fidelity": metrics.trace_fidelity,
                        "recovery_time": metrics.recovery_time
                    }
                    flattened_data.append(row)
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(filepath, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False