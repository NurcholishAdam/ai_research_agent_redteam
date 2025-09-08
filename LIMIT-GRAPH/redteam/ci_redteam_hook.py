# -*- coding: utf-8 -*-
"""
CI Red Team Hook
Automated red team evaluation integration for CI/CD pipeline
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

def setup_ci_logging() -> logging.Logger:
    """Setup logging for CI environment"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ci_redteam.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_redteam_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run red team evaluation with specified configuration
    
    Args:
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        Evaluation results and metadata
    """
    logger = setup_ci_logging()
    logger.info("Starting CI Red Team Evaluation")
    
    # Extract configuration
    strategies = config.get("strategies", ["random", "structural"])
    mask_ratios = config.get("mask_ratios", [0.2, 0.3])
    agent_type = config.get("agent_type", "limit_graph")
    output_file = config.get("output_file", "ci_redteam_results.json")
    
    # Build command
    cmd = [
        sys.executable,
        "redteam_masked_recovery.py",
        "--agent_type", agent_type,
        "--strategies"] + strategies + [
        "--mask_ratios"] + [str(r) for r in mask_ratios] + [
        "--output", output_file,
        "--log_level", "INFO"
    ]
    
    # Add optional parameters
    if config.get("generate_scenarios", False):
        cmd.append("--generate_scenarios")
    
    if "masked_graphs" in config:
        cmd.extend(["--masked_graphs", config["masked_graphs"]])
    
    try:
        # Run evaluation
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=config.get("timeout", 1800)  # 30 minutes default
        )
        
        if result.returncode != 0:
            logger.error(f"Red team evaluation failed: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout
            }
        
        # Load results
        output_path = Path(__file__).parent / output_file
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                evaluation_results = json.load(f)
            
            logger.info(f"Red team evaluation completed successfully")
            logger.info(f"Results saved to {output_path}")
            
            return {
                "success": True,
                "results_file": str(output_path),
                "evaluation_results": evaluation_results,
                "stdout": result.stdout,
                "command": cmd
            }
        else:
            logger.error(f"Results file not found: {output_path}")
            return {
                "success": False,
                "error": f"Results file not found: {output_path}",
                "stdout": result.stdout
            }
    
    except subprocess.TimeoutExpired:
        logger.error("Red team evaluation timed out")
        return {
            "success": False,
            "error": "Evaluation timed out",
            "timeout": config.get("timeout", 1800)
        }
    
    except Exception as e:
        logger.error(f"Error running red team evaluation: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def validate_results(results: Dict[str, Any], 
                    thresholds: Dict[str, float]) -> Dict[str, Any]:
    """
    Validate evaluation results against quality thresholds
    
    Args:
        results: Evaluation results
        thresholds: Quality thresholds for validation
        
    Returns:
        Validation report
    """
    logger = logging.getLogger(__name__)
    
    if not results.get("success", False):
        return {
            "validation_passed": False,
            "reason": "Evaluation failed",
            "details": results.get("error", "Unknown error")
        }
    
    evaluation_results = results.get("evaluation_results", [])
    if not evaluation_results:
        return {
            "validation_passed": False,
            "reason": "No evaluation results found"
        }
    
    # Calculate aggregate metrics
    valid_results = [r for r in evaluation_results if "metrics" in r]
    if not valid_results:
        return {
            "validation_passed": False,
            "reason": "No valid evaluation results"
        }
    
    # Extract metrics
    accuracies = [r["metrics"].accuracy for r in valid_results if hasattr(r["metrics"], 'accuracy')]
    f1_scores = [r["metrics"].f1_score for r in valid_results if hasattr(r["metrics"], 'f1_score')]
    
    if not accuracies or not f1_scores:
        return {
            "validation_passed": False,
            "reason": "No valid metrics found"
        }
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    
    # Check thresholds
    min_accuracy = thresholds.get("min_accuracy", 0.5)
    min_f1_score = thresholds.get("min_f1_score", 0.5)
    min_scenarios = thresholds.get("min_scenarios", 10)
    
    validation_checks = {
        "accuracy_check": avg_accuracy >= min_accuracy,
        "f1_score_check": avg_f1_score >= min_f1_score,
        "scenario_count_check": len(valid_results) >= min_scenarios
    }
    
    validation_passed = all(validation_checks.values())
    
    validation_report = {
        "validation_passed": validation_passed,
        "metrics": {
            "avg_accuracy": avg_accuracy,
            "avg_f1_score": avg_f1_score,
            "scenario_count": len(valid_results),
            "success_rate": len(valid_results) / len(evaluation_results)
        },
        "thresholds": thresholds,
        "checks": validation_checks
    }
    
    if validation_passed:
        logger.info("Red team validation PASSED")
        logger.info(f"Average accuracy: {avg_accuracy:.3f} (threshold: {min_accuracy})")
        logger.info(f"Average F1 score: {avg_f1_score:.3f} (threshold: {min_f1_score})")
    else:
        logger.warning("Red team validation FAILED")
        for check_name, passed in validation_checks.items():
            if not passed:
                logger.warning(f"Failed check: {check_name}")
    
    return validation_report

def update_dashboard(results: Dict[str, Any]) -> bool:
    """
    Update red team dashboard with new results
    
    Args:
        results: Evaluation results
        
    Returns:
        Success status
    """
    logger = logging.getLogger(__name__)
    
    try:
        from redteam.redteam_dashboard import RedTeamDashboard
        
        dashboard = RedTeamDashboard()
        
        if results.get("success", False):
            evaluation_results = results.get("evaluation_results", [])
            dashboard.update_evaluation_data(evaluation_results)
            
            # Export updated dashboard data
            dashboard_export_path = "ci_dashboard_data.json"
            dashboard.export_dashboard_data(dashboard_export_path)
            
            logger.info(f"Dashboard updated with {len(evaluation_results)} results")
            logger.info(f"Dashboard data exported to {dashboard_export_path}")
            
            return True
        else:
            logger.warning("Skipping dashboard update due to evaluation failure")
            return False
    
    except ImportError:
        logger.warning("Dashboard components not available, skipping update")
        return False
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        return False

def generate_ci_report(results: Dict[str, Any], 
                      validation: Dict[str, Any]) -> str:
    """
    Generate CI report for red team evaluation
    
    Args:
        results: Evaluation results
        validation: Validation report
        
    Returns:
        Formatted CI report
    """
    report_lines = [
        "# LIMIT-GRAPH Red Team Evaluation Report",
        f"**Timestamp:** {datetime.now().isoformat()}",
        ""
    ]
    
    # Evaluation Status
    if results.get("success", False):
        report_lines.extend([
            "## ✅ Evaluation Status: SUCCESS",
            ""
        ])
    else:
        report_lines.extend([
            "## ❌ Evaluation Status: FAILED",
            f"**Error:** {results.get('error', 'Unknown error')}",
            ""
        ])
        return "\n".join(report_lines)
    
    # Validation Status
    if validation.get("validation_passed", False):
        report_lines.extend([
            "## ✅ Validation Status: PASSED",
            ""
        ])
    else:
        report_lines.extend([
            "## ❌ Validation Status: FAILED",
            ""
        ])
    
    # Metrics Summary
    metrics = validation.get("metrics", {})
    report_lines.extend([
        "## 📊 Performance Metrics",
        f"- **Average Accuracy:** {metrics.get('avg_accuracy', 0):.3f}",
        f"- **Average F1 Score:** {metrics.get('avg_f1_score', 0):.3f}",
        f"- **Scenarios Evaluated:** {metrics.get('scenario_count', 0)}",
        f"- **Success Rate:** {metrics.get('success_rate', 0):.1%}",
        ""
    ])
    
    # Threshold Checks
    checks = validation.get("checks", {})
    report_lines.extend([
        "## 🎯 Threshold Checks"
    ])
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        report_lines.append(f"- {status} {check_name.replace('_', ' ').title()}")
    
    report_lines.append("")
    
    # Configuration
    thresholds = validation.get("thresholds", {})
    report_lines.extend([
        "## ⚙️ Configuration",
        f"- **Minimum Accuracy Threshold:** {thresholds.get('min_accuracy', 'N/A')}",
        f"- **Minimum F1 Score Threshold:** {thresholds.get('min_f1_score', 'N/A')}",
        f"- **Minimum Scenarios:** {thresholds.get('min_scenarios', 'N/A')}",
        ""
    ])
    
    # Files Generated
    if "results_file" in results:
        report_lines.extend([
            "## 📁 Generated Files",
            f"- **Results:** `{results['results_file']}`",
            f"- **Dashboard Data:** `ci_dashboard_data.json`",
            f"- **Log File:** `ci_redteam.log`",
            ""
        ])
    
    return "\n".join(report_lines)

def main():
    """Main CI hook function"""
    logger = setup_ci_logging()
    
    # Default CI configuration
    default_config = {
        "strategies": ["random", "structural", "adversarial"],
        "mask_ratios": [0.2, 0.3, 0.4],
        "agent_type": "limit_graph",
        "output_file": "ci_redteam_results.json",
        "timeout": 1800,
        "generate_scenarios": False
    }
    
    # Default validation thresholds
    default_thresholds = {
        "min_accuracy": 0.6,
        "min_f1_score": 0.6,
        "min_scenarios": 15
    }
    
    # Load configuration from environment or use defaults
    config = default_config.copy()
    thresholds = default_thresholds.copy()
    
    # Override with environment variables if present
    if "REDTEAM_STRATEGIES" in os.environ:
        config["strategies"] = os.environ["REDTEAM_STRATEGIES"].split(",")
    
    if "REDTEAM_MASK_RATIOS" in os.environ:
        config["mask_ratios"] = [float(x) for x in os.environ["REDTEAM_MASK_RATIOS"].split(",")]
    
    if "REDTEAM_MIN_ACCURACY" in os.environ:
        thresholds["min_accuracy"] = float(os.environ["REDTEAM_MIN_ACCURACY"])
    
    logger.info(f"CI Red Team Configuration: {config}")
    logger.info(f"Validation Thresholds: {thresholds}")
    
    # Run evaluation
    results = run_redteam_evaluation(config)
    
    # Validate results
    validation = validate_results(results, thresholds)
    
    # Update dashboard
    update_dashboard(results)
    
    # Generate CI report
    report = generate_ci_report(results, validation)
    
    # Save report
    report_path = "ci_redteam_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"CI report saved to {report_path}")
    
    # Print report to stdout for CI visibility
    print("\n" + "="*60)
    print(report)
    print("="*60)
    
    # Exit with appropriate code
    if validation.get("validation_passed", False):
        logger.info("CI Red Team Hook completed successfully")
        sys.exit(0)
    else:
        logger.error("CI Red Team Hook failed validation")
        sys.exit(1)

if __name__ == "__main__":
    main()