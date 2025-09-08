# -*- coding: utf-8 -*-
"""
Demo: LIMIT-GRAPH Red Team Masked Recovery System
Demonstrates the complete red team evaluation pipeline
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

from masking_strategy import MaskingStrategy, MaskingType
from recovery_evaluator import RecoveryEvaluator
from masked_recovery_agent import LimitGraphRecoveryAgent, SimpleRecoveryAgent
from redteam_dashboard import RedTeamDashboard

def setup_demo_logging():
    """Setup logging for demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_sample_graphs() -> List[Dict[str, Any]]:
    """Create sample graphs for demonstration"""
    return [
        {
            "edges": [
                {"source": "Alice", "target": "Book", "relation": "owns"},
                {"source": "Bob", "target": "Car", "relation": "drives"},
                {"source": "Alice", "target": "Bob", "relation": "friend"},
                {"source": "Charlie", "target": "House", "relation": "lives_in"}
            ],
            "nodes": ["Alice", "Bob", "Charlie", "Book", "Car", "House"]
        },
        {
            "edges": [
                {"source": "Teacher", "target": "Student", "relation": "teaches"},
                {"source": "Student", "target": "School", "relation": "attends"},
                {"source": "Principal", "target": "School", "relation": "manages"},
                {"source": "Teacher", "target": "Principal", "relation": "reports_to"}
            ],
            "nodes": ["Teacher", "Student", "School", "Principal"]
        },
        {
            "edges": [
                {"source": "Company", "target": "Employee", "relation": "employs"},
                {"source": "Employee", "target": "Project", "relation": "works_on"},
                {"source": "Manager", "target": "Employee", "relation": "supervises"},
                {"source": "CEO", "target": "Company", "relation": "leads"}
            ],
            "nodes": ["Company", "Employee", "Project", "Manager", "CEO"]
        }
    ]

def demo_masking_strategies(logger: logging.Logger):
    """Demonstrate different masking strategies"""
    logger.info("🎭 Demonstrating Masking Strategies")
    
    masking_strategy = MaskingStrategy(seed=42)
    sample_graphs = create_sample_graphs()
    
    strategies = [
        MaskingType.RANDOM,
        MaskingType.STRUCTURAL, 
        MaskingType.ADVERSARIAL,
        MaskingType.SEMANTIC
    ]
    
    for strategy in strategies:
        logger.info(f"\n--- {strategy.value.upper()} MASKING ---")
        
        for i, graph in enumerate(sample_graphs[:2]):  # Use first 2 graphs
            try:
                masked_result = masking_strategy.apply_masking(graph, strategy, 0.3)
                
                logger.info(f"Graph {i+1}:")
                logger.info(f"  Original edges: {len(graph['edges'])}")
                logger.info(f"  Masked edges: {len(masked_result['ground_truth'])}")
                logger.info(f"  Masking ratio: {masked_result['mask_ratio']}")
                
                # Show masked edges
                for gt in masked_result['ground_truth']:
                    logger.info(f"  Masked: {gt['source']} -> [MASK] -> {gt['target']} (was: {gt['original_relation']})")
                
            except Exception as e:
                logger.error(f"Error with {strategy.value} masking on graph {i+1}: {e}")

def demo_agent_recovery(logger: logging.Logger):
    """Demonstrate agent recovery capabilities"""
    logger.info("\n🤖 Demonstrating Agent Recovery")
    
    # Create test scenario
    masking_strategy = MaskingStrategy(seed=42)
    sample_graph = create_sample_graphs()[0]
    
    masked_result = masking_strategy.apply_masking(sample_graph, MaskingType.STRUCTURAL, 0.4)
    
    logger.info("Test Scenario:")
    logger.info(f"  Original graph: {len(sample_graph['edges'])} edges")
    logger.info(f"  Masked edges: {len(masked_result['ground_truth'])}")
    
    # Test different agents
    agents = {
        "Simple Baseline": SimpleRecoveryAgent(),
        "LIMIT-Graph": LimitGraphRecoveryAgent()
    }
    
    query = "Recover the masked relationships in this social network"
    
    for agent_name, agent in agents.items():
        logger.info(f"\n--- {agent_name.upper()} AGENT ---")
        
        try:
            start_time = time.time()
            response = agent.recover_masked_edges(query, masked_result["masked_graph"])
            recovery_time = time.time() - start_time
            
            logger.info(f"Recovery time: {recovery_time:.2f}s")
            logger.info(f"Recovered edges: {len(response.get('recovered_edges', []))}")
            logger.info(f"Confidence: {response.get('reasoning', {}).get('confidence', 'N/A')}")
            
            # Show predictions
            for pred in response.get('predictions', [])[:3]:  # Show first 3
                logger.info(f"  Prediction: {pred['source']} -> {pred['predicted_relation']} -> {pred['target']} (conf: {pred.get('confidence', 'N/A')})")
            
        except Exception as e:
            logger.error(f"Error with {agent_name} agent: {e}")

def demo_evaluation_metrics(logger: logging.Logger):
    """Demonstrate evaluation metrics calculation"""
    logger.info("\n📊 Demonstrating Evaluation Metrics")
    
    evaluator = RecoveryEvaluator()
    
    # Create mock evaluation scenario
    ground_truth = [
        {"original_relation": "owns", "source": "Alice", "target": "Book", "mask_type": "structural"},
        {"original_relation": "friend", "source": "Alice", "target": "Bob", "mask_type": "structural"}
    ]
    
    # Mock agent responses with different quality levels
    agent_responses = {
        "Perfect Agent": {
            "recovered_edges": [
                {"source": "Alice", "target": "Book", "relation": "owns"},
                {"source": "Alice", "target": "Bob", "relation": "friend"}
            ],
            "predictions": [
                {"source": "Alice", "target": "Book", "predicted_relation": "owns", "confidence": 0.95},
                {"source": "Alice", "target": "Bob", "predicted_relation": "friend", "confidence": 0.90}
            ],
            "reasoning": {"confidence": 0.92, "summary": "High confidence predictions"}
        },
        "Partial Agent": {
            "recovered_edges": [
                {"source": "Alice", "target": "Book", "relation": "owns"},
                {"source": "Alice", "target": "Bob", "relation": "colleague"}  # Wrong relation
            ],
            "predictions": [
                {"source": "Alice", "target": "Book", "predicted_relation": "owns", "confidence": 0.85},
                {"source": "Alice", "target": "Bob", "predicted_relation": "colleague", "confidence": 0.60}
            ],
            "reasoning": {"confidence": 0.72, "summary": "Mixed confidence predictions"}
        },
        "Poor Agent": {
            "recovered_edges": [
                {"source": "Alice", "target": "Book", "relation": "dislikes"},  # Wrong
                {"source": "Alice", "target": "Bob", "relation": "enemy"}      # Wrong
            ],
            "predictions": [
                {"source": "Alice", "target": "Book", "predicted_relation": "dislikes", "confidence": 0.40},
                {"source": "Alice", "target": "Bob", "predicted_relation": "enemy", "confidence": 0.35}
            ],
            "reasoning": {"confidence": 0.37, "summary": "Low confidence predictions"}
        }
    }
    
    scenario_metadata = {
        "scenario_id": "demo_001",
        "masking_strategy": "structural",
        "mask_ratio": 0.5
    }
    
    for agent_name, response in agent_responses.items():
        logger.info(f"\n--- {agent_name.upper()} METRICS ---")
        
        try:
            metrics = evaluator.evaluate_recovery(response, ground_truth, scenario_metadata)
            
            logger.info(f"Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"Precision: {metrics.precision:.3f}")
            logger.info(f"Recall: {metrics.recall:.3f}")
            logger.info(f"F1 Score: {metrics.f1_score:.3f}")
            logger.info(f"Confidence Score: {metrics.confidence_score:.3f}")
            logger.info(f"Reasoning Quality: {metrics.reasoning_quality:.3f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {agent_name}: {e}")

def demo_dashboard_visualization(logger: logging.Logger):
    """Demonstrate dashboard capabilities"""
    logger.info("\n📈 Demonstrating Dashboard Visualization")
    
    dashboard = RedTeamDashboard()
    
    # Create mock evaluation data
    mock_evaluations = []
    
    for i in range(20):
        # Simulate different agent performance
        if i < 10:
            # Good agent
            accuracy = 0.8 + (i % 3) * 0.05
            f1_score = 0.75 + (i % 3) * 0.05
            confidence = 0.85 + (i % 2) * 0.05
        else:
            # Weaker agent  
            accuracy = 0.6 + (i % 4) * 0.03
            f1_score = 0.55 + (i % 4) * 0.03
            confidence = 0.65 + (i % 3) * 0.03
        
        from recovery_evaluator import RecoveryMetrics
        
        metrics = RecoveryMetrics(
            accuracy=accuracy,
            precision=accuracy + 0.02,
            recall=accuracy - 0.01,
            f1_score=f1_score,
            confidence_score=confidence,
            reasoning_quality=0.7 + (i % 5) * 0.04,
            trace_fidelity=0.6 + (i % 4) * 0.05,
            recovery_time=1.0 + (i % 3) * 0.5
        )
        
        evaluation = {
            "scenario_id": f"demo_{i:03d}",
            "masking_strategy": ["random", "structural", "adversarial"][i % 3],
            "mask_ratio": [0.2, 0.3, 0.4][i % 3],
            "agent_id": "LIMIT-Graph" if i < 10 else "Baseline",
            "metrics": metrics,
            "timestamp": time.time() - (20 - i) * 3600  # Spread over time
        }
        
        mock_evaluations.append(evaluation)
    
    # Update dashboard
    dashboard.update_evaluation_data(mock_evaluations)
    
    logger.info("Dashboard updated with mock data")
    logger.info(f"Total evaluations: {len(mock_evaluations)}")
    logger.info(f"Leaderboard entries: {len(dashboard.leaderboard_data)}")
    
    # Generate leaderboard
    leaderboard_df = dashboard.create_leaderboard_table()
    logger.info("\nLeaderboard Preview:")
    logger.info(leaderboard_df.to_string(index=False))
    
    # Export dashboard data
    export_path = "demo_dashboard_export.json"
    if dashboard.export_dashboard_data(export_path):
        logger.info(f"Dashboard data exported to {export_path}")

def demo_complete_pipeline(logger: logging.Logger):
    """Demonstrate complete red team pipeline"""
    logger.info("\n🔄 Demonstrating Complete Pipeline")
    
    # 1. Generate scenarios
    logger.info("Step 1: Generating test scenarios")
    masking_strategy = MaskingStrategy(seed=42)
    sample_graphs = create_sample_graphs()
    
    scenarios = masking_strategy.generate_test_scenarios(
        sample_graphs,
        strategies=[MaskingType.RANDOM, MaskingType.STRUCTURAL],
        mask_ratios=[0.2, 0.3]
    )
    
    logger.info(f"Generated {len(scenarios)} scenarios")
    
    # 2. Create agents
    logger.info("Step 2: Creating agents")
    agents = {
        "simple": SimpleRecoveryAgent(),
        "limit_graph": LimitGraphRecoveryAgent()
    }
    
    # 3. Run evaluations
    logger.info("Step 3: Running evaluations")
    evaluator = RecoveryEvaluator()
    all_results = []
    
    for agent_name, agent in agents.items():
        logger.info(f"  Evaluating {agent_name} agent...")
        
        for i, scenario in enumerate(scenarios[:4]):  # Limit for demo
            try:
                query = f"Recover masked relations in scenario {i+1}"
                masked_graph = scenario["masked_graph"]
                ground_truth = scenario["ground_truth"]
                
                # Agent recovery
                response = agent.recover_masked_edges(query, masked_graph)
                
                # Evaluation
                scenario_metadata = {
                    "scenario_id": scenario["scenario_id"],
                    "masking_strategy": scenario["masking_strategy"],
                    "mask_ratio": scenario["mask_ratio"]
                }
                
                metrics = evaluator.evaluate_recovery(response, ground_truth, scenario_metadata)
                
                result = {
                    "agent_id": agent_name,
                    "scenario_id": scenario["scenario_id"],
                    "masking_strategy": scenario["masking_strategy"],
                    "mask_ratio": scenario["mask_ratio"],
                    "metrics": metrics,
                    "timestamp": time.time()
                }
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error in scenario {i+1} for {agent_name}: {e}")
    
    # 4. Generate report
    logger.info("Step 4: Generating evaluation report")
    report = evaluator.generate_evaluation_report()
    
    logger.info(f"Evaluation completed:")
    logger.info(f"  Total scenarios: {report['evaluation_summary']['total_scenarios']}")
    logger.info(f"  Strategies tested: {report['evaluation_summary']['strategies_tested']}")
    
    # Show performance summary
    overall_perf = report.get('overall_performance', {})
    if 'accuracy' in overall_perf:
        logger.info(f"  Average accuracy: {overall_perf['accuracy']['mean']:.3f}")
        logger.info(f"  Average F1 score: {overall_perf['f1_score']['mean']:.3f}")
    
    # 5. Update dashboard
    logger.info("Step 5: Updating dashboard")
    dashboard = RedTeamDashboard()
    dashboard.update_evaluation_data(all_results)
    
    logger.info("✅ Complete pipeline demonstration finished!")

def main():
    """Main demo function"""
    logger = setup_demo_logging()
    
    logger.info("🚀 LIMIT-GRAPH Red Team System Demo")
    logger.info("="*50)
    
    try:
        # Run all demonstrations
        demo_masking_strategies(logger)
        demo_agent_recovery(logger)
        demo_evaluation_metrics(logger)
        demo_dashboard_visualization(logger)
        demo_complete_pipeline(logger)
        
        logger.info("\n🎉 Demo completed successfully!")
        logger.info("Check the generated files for detailed results.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()