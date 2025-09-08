# -*- coding: utf-8 -*-
"""
Red Team Dashboard Module
Visualization and monitoring for masked graph recovery evaluation
"""

import json
import time
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

class RedTeamDashboard:
    """
    Dashboard for visualizing red team evaluation results
    """
    
    def __init__(self):
        """Initialize dashboard"""
        self.evaluation_data = []
        self.leaderboard_data = []
        
    def update_evaluation_data(self, evaluation_results: List[Dict[str, Any]]):
        """Update dashboard with new evaluation results"""
        self.evaluation_data.extend(evaluation_results)
        self._update_leaderboard()
    
    def _update_leaderboard(self):
        """Update leaderboard based on evaluation data"""
        if not self.evaluation_data:
            return
        
        # Aggregate results by agent/model
        agent_performance = {}
        
        for result in self.evaluation_data:
            agent_id = result.get("agent_id", "unknown")
            metrics = result.get("metrics", {})
            
            if agent_id not in agent_performance:
                agent_performance[agent_id] = {
                    "total_scenarios": 0,
                    "accuracy_scores": [],
                    "f1_scores": [],
                    "confidence_scores": [],
                    "reasoning_quality_scores": [],
                    "recovery_times": []
                }
            
            perf = agent_performance[agent_id]
            perf["total_scenarios"] += 1
            
            if hasattr(metrics, 'accuracy'):
                perf["accuracy_scores"].append(metrics.accuracy)
                perf["f1_scores"].append(metrics.f1_score)
                perf["confidence_scores"].append(metrics.confidence_score)
                perf["reasoning_quality_scores"].append(metrics.reasoning_quality)
                perf["recovery_times"].append(metrics.recovery_time)
        
        # Calculate aggregate scores
        leaderboard_entries = []
        for agent_id, perf in agent_performance.items():
            if perf["accuracy_scores"]:
                entry = {
                    "agent_id": agent_id,
                    "scenarios_completed": perf["total_scenarios"],
                    "avg_accuracy": np.mean(perf["accuracy_scores"]),
                    "avg_f1_score": np.mean(perf["f1_scores"]),
                    "avg_confidence": np.mean(perf["confidence_scores"]),
                    "avg_reasoning_quality": np.mean(perf["reasoning_quality_scores"]),
                    "avg_recovery_time": np.mean(perf["recovery_times"]),
                    "overall_score": self._calculate_overall_score(perf),
                    "last_updated": datetime.now().isoformat()
                }
                leaderboard_entries.append(entry)
        
        # Sort by overall score
        self.leaderboard_data = sorted(leaderboard_entries, 
                                     key=lambda x: x["overall_score"], 
                                     reverse=True)
    
    def _calculate_overall_score(self, performance_data: Dict[str, List[float]]) -> float:
        """Calculate overall performance score"""
        weights = {
            "accuracy": 0.3,
            "f1_score": 0.25,
            "confidence": 0.15,
            "reasoning_quality": 0.2,
            "speed": 0.1  # Inverse of recovery time
        }
        
        accuracy = np.mean(performance_data["accuracy_scores"])
        f1_score = np.mean(performance_data["f1_scores"])
        confidence = np.mean(performance_data["confidence_scores"])
        reasoning = np.mean(performance_data["reasoning_quality_scores"])
        
        # Normalize speed (lower time = higher score)
        avg_time = np.mean(performance_data["recovery_times"])
        speed_score = max(0, 1 - (avg_time / 10.0))  # Assume 10s is very slow
        
        overall_score = (
            weights["accuracy"] * accuracy +
            weights["f1_score"] * f1_score +
            weights["confidence"] * confidence +
            weights["reasoning_quality"] * reasoning +
            weights["speed"] * speed_score
        )
        
        return overall_score
    
    def create_performance_overview(self) -> go.Figure:
        """Create performance overview visualization"""
        if not self.evaluation_data:
            return go.Figure().add_annotation(text="No evaluation data available")
        
        # Prepare data
        df_data = []
        for result in self.evaluation_data:
            metrics = result.get("metrics", {})
            if hasattr(metrics, 'accuracy'):
                df_data.append({
                    "Agent": result.get("agent_id", "unknown"),
                    "Strategy": result.get("masking_strategy", "unknown"),
                    "Accuracy": metrics.accuracy,
                    "F1 Score": metrics.f1_score,
                    "Confidence": metrics.confidence_score,
                    "Reasoning Quality": metrics.reasoning_quality,
                    "Recovery Time": metrics.recovery_time
                })
        
        if not df_data:
            return go.Figure().add_annotation(text="No valid metrics data")
        
        df = pd.DataFrame(df_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Accuracy by Strategy", "F1 Score Distribution", 
                          "Confidence vs Accuracy", "Recovery Time by Agent"),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # Accuracy by strategy
        strategy_accuracy = df.groupby("Strategy")["Accuracy"].mean().reset_index()
        fig.add_trace(
            go.Bar(x=strategy_accuracy["Strategy"], y=strategy_accuracy["Accuracy"],
                   name="Avg Accuracy", marker_color="lightblue"),
            row=1, col=1
        )
        
        # F1 Score distribution
        fig.add_trace(
            go.Histogram(x=df["F1 Score"], name="F1 Distribution", 
                        marker_color="lightgreen", nbinsx=20),
            row=1, col=2
        )
        
        # Confidence vs Accuracy scatter
        fig.add_trace(
            go.Scatter(x=df["Confidence"], y=df["Accuracy"], 
                      mode="markers", name="Confidence vs Accuracy",
                      marker=dict(color="red", size=8)),
            row=2, col=1
        )
        
        # Recovery time by agent
        for agent in df["Agent"].unique():
            agent_data = df[df["Agent"] == agent]
            fig.add_trace(
                go.Box(y=agent_data["Recovery Time"], name=agent),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Red Team Evaluation Performance Overview",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_masking_strategy_analysis(self) -> go.Figure:
        """Analyze performance across different masking strategies"""
        if not self.evaluation_data:
            return go.Figure().add_annotation(text="No evaluation data available")
        
        # Prepare strategy performance data
        strategy_data = {}
        for result in self.evaluation_data:
            strategy = result.get("masking_strategy", "unknown")
            metrics = result.get("metrics", {})
            
            if strategy not in strategy_data:
                strategy_data[strategy] = {
                    "accuracy": [],
                    "f1_score": [],
                    "confidence": [],
                    "reasoning_quality": []
                }
            
            if hasattr(metrics, 'accuracy'):
                strategy_data[strategy]["accuracy"].append(metrics.accuracy)
                strategy_data[strategy]["f1_score"].append(metrics.f1_score)
                strategy_data[strategy]["confidence"].append(metrics.confidence_score)
                strategy_data[strategy]["reasoning_quality"].append(metrics.reasoning_quality)
        
        # Create radar chart for each strategy
        strategies = list(strategy_data.keys())
        metrics_names = ["Accuracy", "F1 Score", "Confidence", "Reasoning Quality"]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, strategy in enumerate(strategies):
            if strategy_data[strategy]["accuracy"]:  # Check if data exists
                values = [
                    np.mean(strategy_data[strategy]["accuracy"]),
                    np.mean(strategy_data[strategy]["f1_score"]),
                    np.mean(strategy_data[strategy]["confidence"]),
                    np.mean(strategy_data[strategy]["reasoning_quality"])
                ]
                
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the polygon
                    theta=metrics_names + [metrics_names[0]],
                    fill='toself',
                    name=strategy.title(),
                    line_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Performance by Masking Strategy",
            showlegend=True
        )
        
        return fig
    
    def create_recovery_trace_visualization(self, scenario_id: str) -> go.Figure:
        """Visualize recovery trace for specific scenario"""
        # Find scenario data
        scenario_data = None
        for result in self.evaluation_data:
            if result.get("scenario_id") == scenario_id:
                scenario_data = result
                break
        
        if not scenario_data:
            return go.Figure().add_annotation(text=f"Scenario {scenario_id} not found")
        
        # Extract trace information
        agent_response = scenario_data.get("agent_response", {})
        trace_steps = agent_response.get("trace", [])
        
        if not trace_steps:
            return go.Figure().add_annotation(text="No trace data available")
        
        # Create trace visualization
        fig = go.Figure()
        
        # Add trace steps as timeline
        y_positions = list(range(len(trace_steps)))
        
        fig.add_trace(go.Scatter(
            x=[i for i in range(len(trace_steps))],
            y=y_positions,
            mode='markers+lines+text',
            text=[f"Step {i+1}" for i in range(len(trace_steps))],
            textposition="middle right",
            marker=dict(size=12, color="blue"),
            line=dict(color="lightblue", width=2),
            name="Reasoning Trace"
        ))
        
        # Add step descriptions as annotations
        for i, step in enumerate(trace_steps):
            fig.add_annotation(
                x=i, y=i,
                text=step[:100] + "..." if len(step) > 100 else step,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                ax=50, ay=-30,
                bgcolor="lightyellow",
                bordercolor="gray",
                borderwidth=1
            )
        
        fig.update_layout(
            title=f"Recovery Trace for Scenario {scenario_id}",
            xaxis_title="Reasoning Step",
            yaxis_title="Step Number",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_leaderboard_table(self) -> pd.DataFrame:
        """Create leaderboard table"""
        if not self.leaderboard_data:
            return pd.DataFrame({"Message": ["No leaderboard data available"]})
        
        # Format leaderboard data
        formatted_data = []
        for i, entry in enumerate(self.leaderboard_data):
            formatted_entry = {
                "Rank": i + 1,
                "Agent": entry["agent_id"],
                "Overall Score": f"{entry['overall_score']:.3f}",
                "Accuracy": f"{entry['avg_accuracy']:.3f}",
                "F1 Score": f"{entry['avg_f1_score']:.3f}",
                "Confidence": f"{entry['avg_confidence']:.3f}",
                "Reasoning Quality": f"{entry['avg_reasoning_quality']:.3f}",
                "Avg Time (s)": f"{entry['avg_recovery_time']:.2f}",
                "Scenarios": entry["scenarios_completed"]
            }
            formatted_data.append(formatted_entry)
        
        return pd.DataFrame(formatted_data)
    
    def create_confidence_calibration_plot(self) -> go.Figure:
        """Create confidence calibration plot"""
        if not self.evaluation_data:
            return go.Figure().add_annotation(text="No evaluation data available")
        
        # Extract confidence and accuracy pairs
        confidence_scores = []
        accuracy_scores = []
        
        for result in self.evaluation_data:
            metrics = result.get("metrics", {})
            if hasattr(metrics, 'confidence_score') and hasattr(metrics, 'accuracy'):
                confidence_scores.append(metrics.confidence_score)
                accuracy_scores.append(metrics.accuracy)
        
        if not confidence_scores:
            return go.Figure().add_annotation(text="No confidence data available")
        
        # Create calibration bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this confidence bin
            in_bin = [(conf >= bin_lower) and (conf < bin_upper) 
                     for conf in confidence_scores]
            
            if any(in_bin):
                bin_accuracy = np.mean([acc for acc, in_b in zip(accuracy_scores, in_bin) if in_b])
                bin_confidence = np.mean([conf for conf, in_b in zip(confidence_scores, in_bin) if in_b])
                bin_count = sum(in_bin)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        # Create calibration plot
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=bin_confidences, y=bin_accuracies,
            mode='markers+lines',
            name='Agent Calibration',
            marker=dict(
                size=[count/2 for count in bin_counts],
                color='red',
                sizemode='diameter',
                sizemin=4
            ),
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Confidence Calibration Plot",
            xaxis_title="Mean Predicted Confidence",
            yaxis_title="Mean Actual Accuracy",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            width=600,
            height=600
        )
        
        return fig
    
    def export_dashboard_data(self, filepath: str) -> bool:
        """Export dashboard data for external analysis"""
        try:
            dashboard_export = {
                "evaluation_data": self.evaluation_data,
                "leaderboard_data": self.leaderboard_data,
                "export_timestamp": datetime.now().isoformat(),
                "total_evaluations": len(self.evaluation_data)
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dashboard_export, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting dashboard data: {e}")
            return False

def create_streamlit_dashboard():
    """Create Streamlit dashboard interface"""
    st.set_page_config(
        page_title="LIMIT-GRAPH Red Team Dashboard",
        page_icon="🔴",
        layout="wide"
    )
    
    st.title("🔴 LIMIT-GRAPH Red Team Dashboard")
    st.markdown("Masked Graph Recovery Evaluation Results")
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = RedTeamDashboard()
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    
    # File upload for evaluation data
    uploaded_file = st.sidebar.file_uploader(
        "Upload Evaluation Results", 
        type=['json'],
        help="Upload JSON file with evaluation results"
    )
    
    if uploaded_file is not None:
        try:
            evaluation_data = json.load(uploaded_file)
            dashboard.update_evaluation_data(evaluation_data)
            st.sidebar.success("Data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {e}")
    
    # Main dashboard content
    if dashboard.evaluation_data:
        # Performance overview
        st.header("📊 Performance Overview")
        overview_fig = dashboard.create_performance_overview()
        st.plotly_chart(overview_fig, use_container_width=True)
        
        # Strategy analysis
        st.header("🎯 Masking Strategy Analysis")
        strategy_fig = dashboard.create_masking_strategy_analysis()
        st.plotly_chart(strategy_fig, use_container_width=True)
        
        # Leaderboard
        st.header("🏆 Agent Leaderboard")
        leaderboard_df = dashboard.create_leaderboard_table()
        st.dataframe(leaderboard_df, use_container_width=True)
        
        # Confidence calibration
        st.header("📈 Confidence Calibration")
        calibration_fig = dashboard.create_confidence_calibration_plot()
        st.plotly_chart(calibration_fig, use_container_width=True)
        
        # Scenario trace viewer
        st.header("🔍 Scenario Trace Viewer")
        scenario_ids = [result.get("scenario_id", "unknown") 
                       for result in dashboard.evaluation_data]
        selected_scenario = st.selectbox("Select Scenario", scenario_ids)
        
        if selected_scenario:
            trace_fig = dashboard.create_recovery_trace_visualization(selected_scenario)
            st.plotly_chart(trace_fig, use_container_width=True)
    
    else:
        st.info("Upload evaluation data to view dashboard")
        st.markdown("""
        ### Getting Started
        1. Run red team evaluations using `redteam_masked_recovery.py`
        2. Upload the generated results JSON file
        3. Explore the dashboard visualizations
        """)

if __name__ == "__main__":
    create_streamlit_dashboard()