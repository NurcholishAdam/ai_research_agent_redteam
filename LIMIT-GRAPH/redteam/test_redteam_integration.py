# -*- coding: utf-8 -*-
"""
Integration Tests for LIMIT-GRAPH Red Team Module
Comprehensive testing of masked graph recovery system
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any

from masking_strategy import MaskingStrategy, MaskingType
from recovery_evaluator import RecoveryEvaluator, RecoveryMetrics
from masked_recovery_agent import LimitGraphRecoveryAgent, SimpleRecoveryAgent
from redteam_dashboard import RedTeamDashboard

class TestMaskingStrategy(unittest.TestCase):
    """Test masking strategy functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.masking_strategy = MaskingStrategy(seed=42)
        self.sample_graph = {
            "edges": [
                {"source": "A", "target": "B", "relation": "friend"},
                {"source": "B", "target": "C", "relation": "colleague"},
                {"source": "A", "target": "C", "relation": "neighbor"},
                {"source": "D", "target": "A", "relation": "parent"}
            ],
            "nodes": ["A", "B", "C", "D"]
        }
    
    def test_random_masking(self):
        """Test random masking strategy"""
        result = self.masking_strategy.apply_masking(
            self.sample_graph, MaskingType.RANDOM, 0.5
        )
        
        self.assertIn("masked_graph", result)
        self.assertIn("ground_truth", result)
        self.assertEqual(result["masking_strategy"], "random")
        self.assertEqual(result["mask_ratio"], 0.5)
        
        # Check that some edges are masked
        masked_edges = result["masked_graph"]["edges"]
        mask_count = sum(1 for edge in masked_edges if edge["relation"] == "[MASK]")
        self.assertGreater(mask_count, 0)
        
        # Check ground truth matches masked edges
        self.assertEqual(len(result["ground_truth"]), mask_count)
    
    def test_structural_masking(self):
        """Test structural masking strategy"""
        result = self.masking_strategy.apply_masking(
            self.sample_graph, MaskingType.STRUCTURAL, 0.25
        )
        
        self.assertEqual(result["masking_strategy"], "structural")
        self.assertIn("ground_truth", result)
        
        # Should mask at least one edge
        ground_truth = result["ground_truth"]
        self.assertGreater(len(ground_truth), 0)
        
        # Check mask type is recorded
        for gt in ground_truth:
            self.assertEqual(gt["mask_type"], "structural")
    
    def test_adversarial_masking(self):
        """Test adversarial masking strategy"""
        result = self.masking_strategy.apply_masking(
            self.sample_graph, MaskingType.ADVERSARIAL, 0.3
        )
        
        self.assertEqual(result["masking_strategy"], "adversarial")
        
        # Check for ambiguity level in ground truth
        for gt in result["ground_truth"]:
            self.assertEqual(gt["mask_type"], "adversarial")
            if "ambiguity_level" in gt:
                self.assertIsInstance(gt["ambiguity_level"], int)
    
    def test_generate_test_scenarios(self):
        """Test scenario generation"""
        base_graphs = [self.sample_graph]
        strategies = [MaskingType.RANDOM, MaskingType.STRUCTURAL]
        mask_ratios = [0.2, 0.4]
        
        scenarios = self.masking_strategy.generate_test_scenarios(
            base_graphs, strategies, mask_ratios
        )
        
        # Should generate scenarios for each combination
        expected_count = len(base_graphs) * len(strategies) * len(mask_ratios)
        self.assertEqual(len(scenarios), expected_count)
        
        # Check scenario structure
        for scenario in scenarios:
            self.assertIn("scenario_id", scenario)
            self.assertIn("masked_graph", scenario)
            self.assertIn("ground_truth", scenario)
            self.assertIn("masking_strategy", scenario)
            self.assertIn("mask_ratio", scenario)
            self.assertIn("difficulty_level", scenario)

class TestRecoveryEvaluator(unittest.TestCase):
    """Test recovery evaluation functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.evaluator = RecoveryEvaluator()
        
        self.ground_truth = [
            {"original_relation": "friend", "source": "A", "target": "B", "mask_type": "random"},
            {"original_relation": "colleague", "source": "B", "target": "C", "mask_type": "random"}
        ]
        
        self.scenario_metadata = {
            "scenario_id": "test_001",
            "masking_strategy": "random",
            "mask_ratio": 0.5
        }
    
    def test_perfect_recovery_evaluation(self):
        """Test evaluation with perfect recovery"""
        agent_response = {
            "recovered_edges": [
                {"source": "A", "target": "B", "relation": "friend"},
                {"source": "B", "target": "C", "relation": "colleague"}
            ],
            "predictions": [
                {"source": "A", "target": "B", "predicted_relation": "friend", "confidence": 0.9},
                {"source": "B", "target": "C", "predicted_relation": "colleague", "confidence": 0.85}
            ],
            "reasoning": {"confidence": 0.87}
        }
        
        metrics = self.evaluator.evaluate_recovery(
            agent_response, self.ground_truth, self.scenario_metadata
        )
        
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.precision, 1.0)
        self.assertEqual(metrics.recall, 1.0)
        self.assertEqual(metrics.f1_score, 1.0)
        self.assertGreater(metrics.confidence_score, 0.8)
    
    def test_partial_recovery_evaluation(self):
        """Test evaluation with partial recovery"""
        agent_response = {
            "recovered_edges": [
                {"source": "A", "target": "B", "relation": "friend"},  # Correct
                {"source": "B", "target": "C", "relation": "enemy"}    # Wrong
            ],
            "predictions": [
                {"source": "A", "target": "B", "predicted_relation": "friend", "confidence": 0.8},
                {"source": "B", "target": "C", "predicted_relation": "enemy", "confidence": 0.6}
            ]
        }
        
        metrics = self.evaluator.evaluate_recovery(
            agent_response, self.ground_truth, self.scenario_metadata
        )
        
        self.assertEqual(metrics.accuracy, 0.5)  # 1 out of 2 correct
        self.assertEqual(metrics.precision, 0.5)
        self.assertEqual(metrics.recall, 0.5)
        self.assertAlmostEqual(metrics.f1_score, 0.5, places=2)
    
    def test_no_recovery_evaluation(self):
        """Test evaluation with no recovery"""
        agent_response = {
            "recovered_edges": [],
            "predictions": []
        }
        
        metrics = self.evaluator.evaluate_recovery(
            agent_response, self.ground_truth, self.scenario_metadata
        )
        
        self.assertEqual(metrics.accuracy, 0.0)
        self.assertEqual(metrics.recall, 0.0)
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation"""
        # Add some mock evaluation history
        mock_metrics = RecoveryMetrics(
            accuracy=0.8, precision=0.75, recall=0.85, f1_score=0.8,
            confidence_score=0.7, reasoning_quality=0.6, trace_fidelity=0.5,
            recovery_time=1.2
        )
        
        mock_evaluation = {
            "scenario_id": "test_001",
            "masking_strategy": "random",
            "mask_ratio": 0.3,
            "metrics": mock_metrics,
            "timestamp": 1234567890
        }
        
        self.evaluator.evaluation_history = [mock_evaluation]
        
        report = self.evaluator.generate_evaluation_report()
        
        self.assertIn("evaluation_summary", report)
        self.assertIn("overall_performance", report)
        self.assertEqual(report["evaluation_summary"]["total_scenarios"], 1)

class TestMaskedRecoveryAgent(unittest.TestCase):
    """Test masked recovery agent functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.simple_agent = SimpleRecoveryAgent()
        self.limit_graph_agent = LimitGraphRecoveryAgent()
        
        self.masked_graph = {
            "edges": [
                {"source": "Alice", "target": "Bob", "relation": "[MASK]"},
                {"source": "Charlie", "target": "Dave", "relation": "friend"}
            ],
            "nodes": ["Alice", "Bob", "Charlie", "Dave"]
        }
        
        self.query = "Who is Alice's friend?"
    
    def test_simple_agent_recovery(self):
        """Test simple agent recovery"""
        response = self.simple_agent.recover_masked_edges(
            self.query, self.masked_graph
        )
        
        self.assertIn("recovered_edges", response)
        self.assertIn("predictions", response)
        self.assertIn("reasoning", response)
        
        # Should recover the masked edge
        recovered = response["recovered_edges"]
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0]["source"], "Alice")
        self.assertEqual(recovered[0]["target"], "Bob")
        self.assertNotEqual(recovered[0]["relation"], "[MASK]")
    
    def test_limit_graph_agent_recovery(self):
        """Test LIMIT-Graph agent recovery"""
        response = self.limit_graph_agent.recover_masked_edges(
            self.query, self.masked_graph
        )
        
        self.assertIn("recovered_edges", response)
        self.assertIn("predictions", response)
        self.assertIn("reasoning", response)
        self.assertIn("trace", response)
        
        # Should have reasoning trace
        trace = response["trace"]
        self.assertIsInstance(trace, list)
        self.assertGreater(len(trace), 0)
        
        # Should have confidence scores
        predictions = response["predictions"]
        for pred in predictions:
            self.assertIn("confidence", pred)
            self.assertIsInstance(pred["confidence"], (int, float))
    
    def test_agent_error_handling(self):
        """Test agent error handling with malformed input"""
        malformed_graph = {"edges": "invalid"}
        
        # Should not crash, should return error response
        response = self.limit_graph_agent.recover_masked_edges(
            self.query, malformed_graph
        )
        
        # Should handle gracefully
        self.assertIsInstance(response, dict)

class TestRedTeamDashboard(unittest.TestCase):
    """Test red team dashboard functionality"""
    
    def setUp(self):
        """Setup test fixtures"""
        self.dashboard = RedTeamDashboard()
        
        # Create mock evaluation data
        self.mock_evaluations = []
        for i in range(5):
            metrics = RecoveryMetrics(
                accuracy=0.7 + i * 0.05,
                precision=0.65 + i * 0.05,
                recall=0.75 + i * 0.03,
                f1_score=0.7 + i * 0.04,
                confidence_score=0.6 + i * 0.06,
                reasoning_quality=0.5 + i * 0.08,
                trace_fidelity=0.4 + i * 0.1,
                recovery_time=1.0 + i * 0.2
            )
            
            evaluation = {
                "agent_id": f"agent_{i % 2}",
                "scenario_id": f"scenario_{i}",
                "masking_strategy": ["random", "structural"][i % 2],
                "mask_ratio": 0.2 + i * 0.1,
                "metrics": metrics,
                "timestamp": 1234567890 + i * 3600
            }
            
            self.mock_evaluations.append(evaluation)
    
    def test_update_evaluation_data(self):
        """Test updating dashboard with evaluation data"""
        self.dashboard.update_evaluation_data(self.mock_evaluations)
        
        self.assertEqual(len(self.dashboard.evaluation_data), 5)
        self.assertGreater(len(self.dashboard.leaderboard_data), 0)
    
    def test_create_leaderboard_table(self):
        """Test leaderboard table creation"""
        self.dashboard.update_evaluation_data(self.mock_evaluations)
        
        leaderboard_df = self.dashboard.create_leaderboard_table()
        
        self.assertIsInstance(leaderboard_df, type(self.dashboard.create_leaderboard_table()))
        self.assertGreater(len(leaderboard_df), 0)
    
    def test_export_dashboard_data(self):
        """Test dashboard data export"""
        self.dashboard.update_evaluation_data(self.mock_evaluations)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            success = self.dashboard.export_dashboard_data(temp_path)
            self.assertTrue(success)
            
            # Verify exported data
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIn("evaluation_data", exported_data)
            self.assertIn("leaderboard_data", exported_data)
            self.assertEqual(len(exported_data["evaluation_data"]), 5)
            
        finally:
            os.unlink(temp_path)

class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # 1. Create sample graph
        sample_graph = {
            "edges": [
                {"source": "A", "target": "B", "relation": "friend"},
                {"source": "B", "target": "C", "relation": "colleague"},
                {"source": "A", "target": "C", "relation": "neighbor"}
            ],
            "nodes": ["A", "B", "C"]
        }
        
        # 2. Apply masking
        masking_strategy = MaskingStrategy(seed=42)
        masked_result = masking_strategy.apply_masking(
            sample_graph, MaskingType.RANDOM, 0.33
        )
        
        # 3. Agent recovery
        agent = SimpleRecoveryAgent()
        query = "Recover the masked relationships"
        response = agent.recover_masked_edges(query, masked_result["masked_graph"])
        
        # 4. Evaluate recovery
        evaluator = RecoveryEvaluator()
        scenario_metadata = {
            "scenario_id": "integration_test",
            "masking_strategy": "random",
            "mask_ratio": 0.33
        }
        
        metrics = evaluator.evaluate_recovery(
            response, masked_result["ground_truth"], scenario_metadata
        )
        
        # 5. Update dashboard
        dashboard = RedTeamDashboard()
        evaluation_result = {
            "agent_id": "simple_agent",
            "scenario_id": "integration_test",
            "masking_strategy": "random",
            "mask_ratio": 0.33,
            "metrics": metrics,
            "timestamp": 1234567890
        }
        
        dashboard.update_evaluation_data([evaluation_result])
        
        # Verify workflow completed successfully
        self.assertIsInstance(metrics, RecoveryMetrics)
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertEqual(len(dashboard.evaluation_data), 1)
        self.assertGreater(len(dashboard.leaderboard_data), 0)

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMaskingStrategy,
        TestRecoveryEvaluator,
        TestMaskedRecoveryAgent,
        TestRedTeamDashboard,
        TestIntegrationWorkflow
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🧪 Running LIMIT-GRAPH Red Team Integration Tests")
    print("="*60)
    
    success = run_integration_tests()
    
    if success:
        print("\n✅ All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed!")
        exit(1)