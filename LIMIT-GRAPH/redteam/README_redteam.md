# LIMIT-GRAPH Red Team Module: Masked Graph Recovery

## Overview

The LIMIT-GRAPH Red Team Module evaluates agent robustness through **Masked Graph Recovery** tasks. This module tests whether agents can reconstruct or infer masked semantic relations in graphs, especially those that are critical, ambiguous, or adversarially obfuscated.

## Architecture

### Core Components

1. **Masking Strategy** (`masking_strategy.py`)
   - Implements various masking approaches inspired by SafeProtein's structural masking
   - Supports random, structural, critical path, adversarial, and semantic masking
   - Generates test scenarios with different difficulty levels

2. **Recovery Evaluator** (`recovery_evaluator.py`)
   - Evaluates agent performance on masked recovery tasks
   - Calculates accuracy, precision, recall, F1-score, confidence, reasoning quality, and trace fidelity
   - Generates comprehensive evaluation reports

3. **Masked Recovery Agent** (`masked_recovery_agent.py`)
   - Abstract interface for recovery agents
   - LIMIT-GRAPH implementation using graph reasoning and entity linking
   - Simple baseline agent for comparison

4. **Red Team Dashboard** (`redteam_dashboard.py`)
   - Visualization and monitoring interface
   - Performance analytics and leaderboard
   - Confidence calibration plots and trace visualization

## Masking Strategies

### 1. Random Masking
- Randomly selects edges to mask
- Baseline difficulty level
- Tests general recovery capabilities

### 2. Structural Masking
- Targets high-degree nodes and central edges
- Based on graph topology analysis
- Tests understanding of graph structure

### 3. Critical Path Masking
- Masks edges on shortest paths between important nodes
- Tests path reasoning capabilities
- Higher difficulty due to connectivity importance

### 4. Adversarial Masking
- Creates ambiguous scenarios with multiple possible relations
- Targets relations with similar semantic types
- Tests disambiguation and reasoning under uncertainty

### 5. Semantic Masking
- Groups relations by semantic similarity
- Distributes masking across semantic categories
- Tests semantic understanding and categorization

## Evaluation Metrics

### Core Metrics
- **Accuracy**: Exact match recovery rate
- **Precision**: Correct recoveries / total recoveries
- **Recall**: Correct recoveries / total expected
- **F1-Score**: Harmonic mean of precision and recall

### Advanced Metrics
- **Confidence Score**: Agent's confidence calibration
- **Reasoning Quality**: Quality of reasoning process and explanations
- **Trace Fidelity**: Accuracy and completeness of reasoning traces
- **Recovery Time**: Processing time for recovery tasks

## Usage

### Basic Evaluation

```bash
# Run evaluation with default settings
python redteam_masked_recovery.py

# Specify agent type and strategies
python redteam_masked_recovery.py --agent_type limit_graph --strategies random structural adversarial

# Generate new test scenarios
python redteam_masked_recovery.py --generate_scenarios --mask_ratios 0.2 0.3 0.5
```

### Advanced Configuration

```bash
# Custom masked graphs file
python redteam_masked_recovery.py --masked_graphs custom_scenarios.jsonl

# Multiple masking strategies with different ratios
python redteam_masked_recovery.py \
  --strategies random structural critical_path adversarial semantic \
  --mask_ratios 0.1 0.2 0.3 0.4 0.5 \
  --output detailed_evaluation.json

# Debug mode with verbose logging
python redteam_masked_recovery.py --log_level DEBUG
```

### Dashboard Usage

```bash
# Launch Streamlit dashboard
streamlit run redteam_dashboard.py

# Or use programmatically
python -c "
from redteam.redteam_dashboard import RedTeamDashboard
dashboard = RedTeamDashboard()
# Load and visualize results
"
```

## Data Format

### Masked Graph Scenario Format

```json
{
  "scenario_id": "unique_identifier",
  "query": "Natural language query",
  "masked_graph": {
    "edges": [
      {"source": "Entity1", "target": "Entity2", "relation": "[MASK]"},
      {"source": "Entity3", "target": "Entity4", "relation": "known_relation"}
    ],
    "nodes": ["Entity1", "Entity2", "Entity3", "Entity4"]
  },
  "ground_truth": [
    {
      "original_relation": "expected_relation",
      "source": "Entity1",
      "target": "Entity2",
      "mask_type": "masking_strategy_used"
    }
  ],
  "masking_strategy": "strategy_name",
  "mask_ratio": 0.3,
  "difficulty_level": "easy|medium|hard|expert",
  "lang": "en|id"
}
```

### Agent Response Format

```json
{
  "query": "Original query",
  "recovered_edges": [
    {
      "source": "Entity1",
      "target": "Entity2", 
      "relation": "predicted_relation"
    }
  ],
  "predictions": [
    {
      "source": "Entity1",
      "target": "Entity2",
      "predicted_relation": "predicted_relation",
      "confidence": 0.85,
      "alternatives": ["alt1", "alt2", "alt3"]
    }
  ],
  "reasoning": {
    "approach": "strategy_description",
    "confidence": 0.85,
    "summary": "reasoning_summary"
  },
  "trace": ["step1", "step2", "step3"],
  "processing_time": 1.23
}
```

## Integration with LIMIT-GRAPH

### CI/CD Integration

The red team module integrates with LIMIT-GRAPH's CI workflow:

```yaml
# .github/workflows/limit-graph-ci.yml
- name: Run Red Team Evaluation
  run: |
    python extensions/LIMIT-GRAPH/CI_Workflow/redteam_masked_recovery.py \
      --strategies random structural adversarial \
      --output ci_redteam_results.json
    
- name: Update Dashboard
  run: |
    python -c "
    from extensions.LIMIT-GRAPH.redteam.redteam_dashboard import RedTeamDashboard
    dashboard = RedTeamDashboard()
    # Update with CI results
    "
```

### Dashboard Integration

The dashboard provides real-time monitoring of:

- **Masked vs Recovered Edges**: Visual comparison of predictions vs ground truth
- **Reasoning Path Overlays**: Step-by-step reasoning visualization
- **Confidence Score Calibration**: Confidence vs accuracy analysis
- **Agent Leaderboard**: Performance ranking across different strategies
- **Strategy Performance**: Comparative analysis across masking strategies

## Leaderboard Format

The red team leaderboard tracks:

| Rank | Agent | Overall Score | Accuracy | F1 Score | Confidence | Reasoning Quality | Avg Time | Scenarios |
|------|-------|---------------|----------|----------|------------|-------------------|----------|-----------|
| 1 | LIMIT-Graph-v2 | 0.847 | 0.892 | 0.876 | 0.834 | 0.823 | 2.1s | 150 |
| 2 | LIMIT-Graph-v1 | 0.782 | 0.821 | 0.798 | 0.756 | 0.734 | 3.2s | 150 |
| 3 | Baseline | 0.456 | 0.523 | 0.489 | 0.412 | 0.387 | 0.8s | 150 |

## Development

### Adding New Masking Strategies

1. Extend `MaskingType` enum in `masking_strategy.py`
2. Implement strategy method in `MaskingStrategy` class
3. Update CLI arguments and documentation

### Custom Agent Implementation

```python
from redteam.masked_recovery_agent import MaskedRecoveryAgent

class CustomAgent(MaskedRecoveryAgent):
    def recover_masked_edges(self, query, masked_graph, context=None):
        # Implement custom recovery logic
        return {
            "recovered_edges": [...],
            "predictions": [...],
            "reasoning": {...}
        }
```

### Custom Evaluation Metrics

```python
from redteam.recovery_evaluator import RecoveryEvaluator

class CustomEvaluator(RecoveryEvaluator):
    def _calculate_custom_metric(self, agent_response, ground_truth):
        # Implement custom evaluation logic
        return metric_value
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure LIMIT-GRAPH components are properly installed
2. **Memory Issues**: Reduce batch size for large scenario sets
3. **Timeout Errors**: Increase timeout for complex reasoning tasks

### Debug Mode

```bash
python redteam_masked_recovery.py --log_level DEBUG
```

This provides detailed logging of:
- Scenario loading and processing
- Agent reasoning steps
- Evaluation calculations
- Error traces

## Contributing

1. Follow the existing code structure and documentation standards
2. Add comprehensive tests for new components
3. Update documentation and examples
4. Ensure compatibility with existing LIMIT-GRAPH infrastructure

## License

This module is part of the LIMIT-GRAPH project and follows the same licensing terms.