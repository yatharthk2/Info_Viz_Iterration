"""
Model Monitoring Utilities for CPI Analysis & Prediction Dashboard.

This module provides functions for tracking model performance over time,
detecting model drift, and visualizing performance metrics.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import json
import os
import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

# Import visualization utilities for consistent styling
from utils.visualization import format_chart_for_dark_mode
from utils.theme import COLOR_SYSTEM, TYPOGRAPHY

# Configure logging
logger = logging.getLogger(__name__)

# Define constants
MONITORING_FILE = "model_monitoring_data.json"
DEFAULT_METRICS = ["mae", "rmse", "r2"]

class ModelMonitor:
    """Class for monitoring model drift and performance over time."""
    
    def __init__(self, model_name: str):
        """
        Initialize the model monitor.
        
        Args:
            model_name (str): Name of the model being monitored
        """
        self.model_name = model_name
        self.monitoring_data = self._load_monitoring_data()
    
    def _load_monitoring_data(self) -> Dict[str, Any]:
        """
        Load existing monitoring data or create new structure if none exists.
        
        Returns:
            Dict[str, Any]: Dictionary containing monitoring data
        """
        try:
            if os.path.exists(MONITORING_FILE):
                with open(MONITORING_FILE, 'r') as f:
                    data = json.load(f)
                    if self.model_name not in data:
                        data[self.model_name] = {
                            "metrics": [],
                            "drift_alerts": [],
                            "last_updated": None
                        }
                    return data
            else:
                # Create new monitoring data structure
                return {
                    self.model_name: {
                        "metrics": [],
                        "drift_alerts": [],
                        "last_updated": None
                    }
                }
        except Exception as e:
            logger.error(f"Error loading monitoring data: {e}")
            # Return empty structure in case of error
            return {
                self.model_name: {
                    "metrics": [],
                    "drift_alerts": [],
                    "last_updated": None
                }
            }
    
    def _save_monitoring_data(self) -> None:
        """Save monitoring data to file."""
        try:
            with open(MONITORING_FILE, 'w') as f:
                json.dump(self.monitoring_data, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], date: Optional[str] = None) -> None:
        """
        Log model performance metrics for a specific date.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics (e.g., {'mae': 2.5})
            date (Optional[str]): Date string in YYYY-MM-DD format, defaults to today
        """
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        metric_entry = {
            "date": date,
            "metrics": metrics
        }
        
        self.monitoring_data[self.model_name]["metrics"].append(metric_entry)
        self.monitoring_data[self.model_name]["last_updated"] = date
        self._save_monitoring_data()
        logger.info(f"Logged metrics for {self.model_name} on {date}")
    
    def log_drift_alert(self, metric: str, threshold: float, 
                        current_value: float, previous_value: float,
                        date: Optional[str] = None) -> None:
        """
        Log a drift alert when metrics exceed a threshold.
        
        Args:
            metric (str): Name of the metric showing drift
            threshold (float): Threshold that was exceeded
            current_value (float): Current metric value
            previous_value (float): Previous metric value
            date (Optional[str]): Date string in YYYY-MM-DD format, defaults to today
        """
        if date is None:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        alert = {
            "date": date,
            "metric": metric,
            "threshold": threshold,
            "current_value": current_value,
            "previous_value": previous_value,
            "percent_change": ((current_value - previous_value) / previous_value * 100)
                              if previous_value != 0 else float('inf')
        }
        
        self.monitoring_data[self.model_name]["drift_alerts"].append(alert)
        self._save_monitoring_data()
        logger.warning(f"Drift alert logged for {self.model_name} on {date}: {metric}")
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get the history of all logged metrics as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with metrics history
        """
        metrics_data = self.monitoring_data[self.model_name]["metrics"]
        if not metrics_data:
            return pd.DataFrame()
        
        # Flatten metrics data for DataFrame
        rows = []
        for entry in metrics_data:
            date = entry["date"]
            for metric_name, value in entry["metrics"].items():
                rows.append({
                    "date": date,
                    "metric": metric_name,
                    "value": value
                })
        
        return pd.DataFrame(rows)
    
    def get_alerts(self) -> pd.DataFrame:
        """
        Get all drift alerts as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with drift alerts
        """
        alerts = self.monitoring_data[self.model_name]["drift_alerts"]
        if not alerts:
            return pd.DataFrame()
        
        return pd.DataFrame(alerts)
    
    def detect_drift(self, new_metrics: Dict[str, float], 
                     threshold_pct: float = 10.0) -> List[Dict[str, Any]]:
        """
        Detect if new metrics indicate model drift compared to historical data.
        
        Args:
            new_metrics (Dict[str, float]): New metrics to compare
            threshold_pct (float): Percentage threshold for drift detection
            
        Returns:
            List[Dict[str, Any]]: List of drift alerts if any
        """
        metrics_df = self.get_metrics_history()
        alerts = []
        
        if metrics_df.empty or len(metrics_df) < 2:
            # Not enough historical data for comparison
            return []
        
        for metric_name, new_value in new_metrics.items():
            # Get historical values for this metric
            metric_history = metrics_df[metrics_df["metric"] == metric_name]
            
            if len(metric_history) < 2:
                continue
                
            # Get last recorded value (excluding the most recent)
            previous_values = metric_history.sort_values("date")["value"].values
            if len(previous_values) <= 1:
                continue
                
            previous_value = previous_values[-2]  # Second to last value
            
            # Calculate percent change
            percent_change = abs((new_value - previous_value) / previous_value * 100) if previous_value != 0 else float('inf')
            
            # Check if change exceeds threshold
            if percent_change > threshold_pct:
                alert = {
                    "metric": metric_name,
                    "threshold": threshold_pct,
                    "current_value": new_value,
                    "previous_value": previous_value,
                    "percent_change": percent_change
                }
                alerts.append(alert)
                
                # Log the alert
                self.log_drift_alert(
                    metric=metric_name,
                    threshold=threshold_pct,
                    current_value=new_value,
                    previous_value=previous_value
                )
        
        return alerts

def create_metrics_trend_chart(metrics_df: pd.DataFrame, metric_names: Optional[List[str]] = None) -> go.Figure:
    """
    Create a line chart showing trends in model metrics over time.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame with metrics history
        metric_names (List[str]): List of metrics to include, defaults to all
        
    Returns:
        go.Figure: Plotly figure with metrics trends
    """
    if metrics_df.empty:
        # Create an empty chart with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics data available yet",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=COLOR_SYSTEM['TEXT']['PRIMARY'])
        )
        return format_chart_for_dark_mode(fig, "Model Metrics Trends", height=400)
    
    # Filter by selected metrics if provided
    if metric_names:
        filtered_df = metrics_df[metrics_df["metric"].isin(metric_names)]
    else:
        filtered_df = metrics_df
    
    if filtered_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No matching metrics found",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=COLOR_SYSTEM['TEXT']['PRIMARY'])
        )
        return format_chart_for_dark_mode(fig, "Model Metrics Trends", height=400)
    
    # Create figure with a separate axis for each metric
    fig = go.Figure()
    
    # Get unique metrics and assign colors
    unique_metrics = filtered_df["metric"].unique()
    colors = px.colors.qualitative.Bold[:len(unique_metrics)]
    color_map = dict(zip(unique_metrics, colors))
    
    # Add each metric as a separate line
    for metric in unique_metrics:
        metric_data = filtered_df[filtered_df["metric"] == metric]
        
        fig.add_trace(go.Scatter(
            x=metric_data["date"],
            y=metric_data["value"],
            mode="lines+markers",
            name=metric.upper(),
            line=dict(color=color_map[metric], width=2),
            marker=dict(size=8, color=color_map[metric]),
            hovertemplate=(
                "<b>%{x}</b><br>" +
                f"{metric.upper()}: %{{y:.4f}}<br>" +
                "<extra></extra>"
            )
        ))
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Model Metrics Over Time", height=400)
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Metric Value",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_drift_alerts_table(alerts_df: pd.DataFrame) -> go.Figure:
    """
    Create a table showing model drift alerts.
    
    Args:
        alerts_df (pd.DataFrame): DataFrame with drift alerts
        
    Returns:
        go.Figure: Plotly figure with drift alerts table
    """
    if alerts_df.empty:
        # Create an empty chart with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No drift alerts detected",
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color=COLOR_SYSTEM['TEXT']['PRIMARY'])
        )
        return format_chart_for_dark_mode(fig, "Model Drift Alerts", height=300)
    
    # Format percent change values
    alerts_df["Change"] = alerts_df["percent_change"].apply(lambda x: f"{x:.2f}%")
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Date", "Metric", "Previous", "Current", "Change"],
            fill_color=COLOR_SYSTEM['BACKGROUND']['SECONDARY'],
            align="left",
            font=dict(color=COLOR_SYSTEM['TEXT']['PRIMARY'], size=14)
        ),
        cells=dict(
            values=[
                alerts_df["date"],
                alerts_df["metric"].str.upper(),
                alerts_df["previous_value"].round(4),
                alerts_df["current_value"].round(4),
                alerts_df["Change"]
            ],
            fill_color=COLOR_SYSTEM['BACKGROUND']['PRIMARY'],
            align="left",
            font=dict(color=COLOR_SYSTEM['TEXT']['PRIMARY'], size=12),
            height=30
        )
    )])
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Model Drift Alerts", height=len(alerts_df) * 30 + 50)
    
    return fig

def show_model_monitoring(model_name: str = "cpi_prediction") -> None:
    """
    Display model monitoring dashboard in Streamlit.
    
    Args:
        model_name (str): Name of the model to monitor
    """
    st.title("Model Drift Monitoring")
    
    # Create model monitor instance
    monitor = ModelMonitor(model_name)
    
    # Get metrics history
    metrics_df = monitor.get_metrics_history()
    alerts_df = monitor.get_alerts()
    
    # Add a brief explanation
    st.markdown("""
    This section tracks the CPI prediction model's performance over time to ensure predictions 
    remain accurate. Significant changes in performance metrics could indicate model drift, 
    which may require retraining.
    """)
    
    # Display key monitoring stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_records = len(metrics_df["date"].unique()) if not metrics_df.empty else 0
        st.metric("Performance Records", num_records)
    
    with col2:
        num_alerts = len(alerts_df) if not alerts_df.empty else 0
        st.metric("Drift Alerts", num_alerts, 
                 delta=f"+{num_alerts}" if num_alerts > 0 else "0",
                 delta_color="inverse")
    
    with col3:
        last_updated = monitor.monitoring_data[model_name]["last_updated"]
        if last_updated:
            st.metric("Last Updated", last_updated)
        else:
            st.metric("Last Updated", "Never")
    
    # Metrics trend chart
    st.subheader("Performance Metrics Trends")
    
    # Add metric selection
    if not metrics_df.empty:
        available_metrics = sorted(metrics_df["metric"].unique())
        selected_metrics = st.multiselect(
            "Select metrics to display",
            options=available_metrics,
            default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics
        )
    else:
        selected_metrics = []
    
    metrics_chart = create_metrics_trend_chart(metrics_df, selected_metrics)
    st.plotly_chart(metrics_chart, use_container_width=True)
    
    # Drift alerts
    st.subheader("Model Drift Alerts")
    if not alerts_df.empty:
        # Add filtering options
        date_range = st.date_input(
            "Filter by date range",
            value=(
                pd.to_datetime(alerts_df["date"].min()),
                pd.to_datetime(alerts_df["date"].max())
            ) if not alerts_df.empty else (None, None),
            disabled=alerts_df.empty
        )
        
        if len(date_range) == 2 and not alerts_df.empty:
            start_date, end_date = date_range
            filtered_alerts = alerts_df[
                (pd.to_datetime(alerts_df["date"]) >= start_date) &
                (pd.to_datetime(alerts_df["date"]) <= end_date)
            ]
        else:
            filtered_alerts = alerts_df
    else:
        filtered_alerts = alerts_df
    
    alerts_table = create_drift_alerts_table(filtered_alerts)
    st.plotly_chart(alerts_table, use_container_width=True)
    
    # Recommendations based on monitoring
    st.subheader("Recommendations")
    if num_alerts > 3:
        st.warning("""
        **High Alert:** Multiple drift indicators detected. Consider retraining the model
        with more recent data to maintain prediction accuracy.
        """)
    elif num_alerts > 0:
        st.info("""
        **Moderate Alert:** Some drift indicators detected. Monitor closely and consider
        validation with recent data.
        """)
    else:
        st.success("""
        **Good Status:** No significant model drift detected. Continue monitoring.
        """)
    
    # Add manual testing section
    with st.expander("Manual Performance Testing"):
        st.markdown("""
        Use this section to log new performance metrics after testing the model on recent data.
        This helps track model performance over time.
        """)
        
        with st.form("log_metrics_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                mae = st.number_input("Mean Absolute Error (MAE)", min_value=0.0, format="%.4f")
                rmse = st.number_input("Root Mean Squared Error (RMSE)", min_value=0.0, format="%.4f")
            
            with col2:
                r2 = st.number_input("R² Score", min_value=-1.0, max_value=1.0, format="%.4f")
                test_date = st.date_input("Test Date", value=datetime.datetime.now())
            
            submit = st.form_submit_button("Log Metrics")
            
            if submit:
                # Collect metrics
                new_metrics = {
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                }
                
                # Log metrics
                monitor.log_metrics(new_metrics, date=test_date.strftime("%Y-%m-%d"))
                
                # Check for drift
                drift_alerts = monitor.detect_drift(new_metrics)
                
                if drift_alerts:
                    st.warning(f"⚠️ Detected potential model drift in {len(drift_alerts)} metrics!")
                else:
                    st.success("✅ Metrics logged successfully. No significant drift detected.")
                
                # Refresh the page to show new data
                st.rerun()