"""
Model metrics utilities for enhanced model evaluation and explainability.
This module provides functionality to evaluate model performance,
detect overfitting/underfitting, and visualize metrics.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import logging
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# Removed matplotlib dependency in favor of Plotly
import io
import base64

# Import visualization utilities
from utils.theme import COLOR_SYSTEM, format_chart_for_dark_mode, TYPOGRAPHY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model_performance(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                              y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate model performance with comprehensive metrics.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dict with evaluation metrics including train/test comparisons for overfitting detection
    """
    results = {}
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Basic metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Store all metrics
    results['MSE'] = {'train': train_mse, 'test': test_mse}
    results['RMSE'] = {'train': train_rmse, 'test': test_rmse}
    results['MAE'] = {'train': train_mae, 'test': test_mae}
    results['R²'] = {'train': train_r2, 'test': test_r2}
    
    # Calculate overfitting metrics
    results['overfitting_ratio'] = (train_r2 - test_r2) / train_r2 if train_r2 > 0 else 0
    results['generalization_error'] = test_rmse - train_rmse
    
    # Assess overfitting/underfitting
    if results['overfitting_ratio'] > 0.2:
        results['fitting_assessment'] = 'Overfitting'
    elif results['overfitting_ratio'] < 0:
        results['fitting_assessment'] = 'Underfitting'
    else:
        results['fitting_assessment'] = 'Good fit'
    
    # Add interpretation
    if results['fitting_assessment'] == 'Overfitting':
        results['recommendation'] = (
            "Model is likely overfitting. Consider: \n"
            "1. Regularization techniques\n"
            "2. Reducing model complexity\n"
            "3. Feature selection\n"
            "4. More training data"
        )
    elif results['fitting_assessment'] == 'Underfitting':
        results['recommendation'] = (
            "Model is likely underfitting. Consider: \n"
            "1. Increasing model complexity\n"
            "2. Adding more features\n"
            "3. Reducing regularization\n"
            "4. Trying different algorithms"
        )
    else:
        results['recommendation'] = "Model has a good balance between bias and variance."
    
    return results

def plot_learning_curve(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> go.Figure:
    """
    Generate a learning curve visualization to detect overfitting/underfitting.
    
    Args:
        model: Model to evaluate
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds
        
    Returns:
        Plotly figure with learning curve
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # Add training scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=train_scores_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color=COLOR_SYSTEM['ACCENT']['BLUE'], width=2),
        marker=dict(size=8, color=COLOR_SYSTEM['ACCENT']['BLUE']),
        hovertemplate='Sample size: %{x}<br>Score: %{y:.4f}<extra></extra>'
    ))
    
    # Add confidence interval for training
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([
            train_scores_mean + train_scores_std,
            (train_scores_mean - train_scores_std)[::-1]
        ]),
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(78, 121, 167, 0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add test scores
    fig.add_trace(go.Scatter(
        x=train_sizes,
        y=test_scores_mean,
        mode='lines+markers',
        name='Cross-Validation Score',
        line=dict(color=COLOR_SYSTEM['ACCENT']['ORANGE'], width=2),
        marker=dict(size=8, color=COLOR_SYSTEM['ACCENT']['ORANGE']),
        hovertemplate='Sample size: %{x}<br>Score: %{y:.4f}<extra></extra>'
    ))
    
    # Add confidence interval for test
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([
            test_scores_mean + test_scores_std,
            (test_scores_mean - test_scores_std)[::-1]
        ]),
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(242, 142, 43, 0.2)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add a gap annotation for overfitting
    gap = train_scores_mean[-1] - test_scores_mean[-1]
    gap_color = COLOR_SYSTEM['SEMANTIC']['ERROR'] if gap > 0.1 else COLOR_SYSTEM['SEMANTIC']['SUCCESS']
    
    # Add annotation for the gap
    if gap > 0.05:
        fig.add_annotation(
            x=train_sizes[-1],
            y=(train_scores_mean[-1] + test_scores_mean[-1]) / 2,
            text=f"Gap: {gap:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=gap_color,
            font=dict(size=12, color=gap_color)
        )
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Learning Curve Analysis")
    
    fig.update_layout(
        xaxis_title="Training Examples",
        yaxis_title="R² Score",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add annotation for interpretation
    if train_scores_mean[-1] > 0.9 and test_scores_mean[-1] < 0.7:
        diagnostic = "Overfitting"
        advice = "Reduce model complexity or regularize"
    elif train_scores_mean[-1] < 0.7 and test_scores_mean[-1] < 0.6:
        diagnostic = "Underfitting"
        advice = "Increase model complexity or add features"
    elif train_scores_mean[-1] - test_scores_mean[-1] > 0.1:
        diagnostic = "Slight Overfitting"
        advice = "Consider regularization or more data"
    else:
        diagnostic = "Good Fit"
        advice = "Model is well balanced"
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Diagnosis: {diagnostic}<br>{advice}",
        showarrow=False,
        font=dict(size=12, color=COLOR_SYSTEM['PRIMARY']['CONTRAST']),
        bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
        bordercolor=COLOR_SYSTEM['NEUTRAL']['DARK'],
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig

def create_residuals_plot(model: Any, X: pd.DataFrame, y: pd.Series) -> go.Figure:
    """
    Create a residuals plot to evaluate model fit quality.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        
    Returns:
        Plotly figure with residuals visualization
    """
    # Get predictions
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot of residuals
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(
            size=8,
            color=COLOR_SYSTEM['ACCENT']['BLUE'],
            opacity=0.6,
            line=dict(width=1, color=COLOR_SYSTEM['NEUTRAL']['DARKEST'])
        ),
        name='Residuals',
        hovertemplate=(
            "Predicted CPI: $%{x:.2f}<br>" +
            "Residual: $%{y:.2f}<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=min(y_pred),
        x1=max(y_pred),
        y0=0,
        y1=0,
        line=dict(
            color=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
            width=2,
            dash="dash",
        )
    )
    
    # Calculate outlier threshold (1.5 * IQR)
    q1 = np.percentile(residuals, 25)
    q3 = np.percentile(residuals, 75)
    iqr = q3 - q1
    outlier_threshold = 1.5 * iqr
    
    # Highlight outliers
    outlier_indices = np.where((residuals > q3 + outlier_threshold) | (residuals < q1 - outlier_threshold))[0]
    if len(outlier_indices) > 0:
        fig.add_trace(go.Scatter(
            x=y_pred[outlier_indices],
            y=residuals[outlier_indices],
            mode='markers',
            marker=dict(
                size=12,
                color=COLOR_SYSTEM['SEMANTIC']['ERROR'],
                symbol='circle-open',
                line=dict(width=2, color=COLOR_SYSTEM['SEMANTIC']['ERROR'])
            ),
            name='Outliers',
            hovertemplate=(
                "Predicted CPI: $%{x:.2f}<br>" +
                "Residual: $%{y:.2f}<br>" +
                "<extra>Potential outlier</extra>"
            )
        ))
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Residuals Analysis")
    
    fig.update_layout(
        xaxis_title="Predicted CPI ($)",
        yaxis_title="Residuals ($)",
        hovermode="closest"
    )
    
    # Residuals statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Add annotations for residual statistics
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        text=f"Mean: ${mean_residual:.2f}<br>Std Dev: ${std_residual:.2f}",
        showarrow=False,
        font=dict(size=12, color=COLOR_SYSTEM['PRIMARY']['CONTRAST']),
        bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
        bordercolor=COLOR_SYSTEM['NEUTRAL']['DARK'],
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig

def create_feature_dependence_plot(model: Any, X: pd.DataFrame, feature_name: str) -> go.Figure:
    """
    Create a partial dependence plot for a specific feature.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_name: Name of the feature to analyze
        
    Returns:
        Plotly figure with feature dependence visualization
    """
    if feature_name not in X.columns:
        raise ValueError(f"Feature '{feature_name}' not found in dataset")
    
    # Create a range of values for the feature
    feature_min = X[feature_name].min()
    feature_max = X[feature_name].max()
    feature_range = np.linspace(feature_min, feature_max, 50)
    
    # Initialize predictions array
    predictions = []
    
    # For each value in the range, predict using the model
    for value in feature_range:
        X_copy = X.copy()
        X_copy[feature_name] = value
        pred = model.predict(X_copy).mean()
        predictions.append(pred)
    
    # Create figure
    fig = go.Figure()
    
    # Add line plot
    fig.add_trace(go.Scatter(
        x=feature_range,
        y=predictions,
        mode='lines',
        line=dict(color=COLOR_SYSTEM['ACCENT']['BLUE'], width=3),
        name='Predicted CPI',
        hovertemplate=(
            f"{feature_name}: %{{x:.2f}}<br>" +
            "Predicted CPI: $%{y:.2f}<br>" +
            "<extra></extra>"
        )
    ))
    
    # Add distribution of actual feature values as histogram
    feature_counts, feature_bins = np.histogram(X[feature_name], bins=20, density=True)
    bin_centers = 0.5 * (feature_bins[:-1] + feature_bins[1:])
    
    # Scale the histogram to fit nicely on the chart
    y_range = max(predictions) - min(predictions)
    scale_factor = y_range * 0.2 / max(feature_counts) if max(feature_counts) > 0 else 0
    scaled_counts = feature_counts * scale_factor + min(predictions)
    
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=scaled_counts,
        marker=dict(
            color=COLOR_SYSTEM['NEUTRAL']['MEDIUM'],
            opacity=0.3
        ),
        name='Feature Distribution',
        hovertemplate=(
            f"{feature_name}: %{{x:.2f}}<br>" +
            "Frequency: %{customdata:.2f}<br>" +
            "<extra></extra>"
        ),
        customdata=feature_counts
    ))
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, f"Impact of {feature_name} on CPI")
    
    fig.update_layout(
        xaxis_title=feature_name,
        yaxis_title="Predicted CPI ($)",
        hovermode="closest"
    )
    
    # Determine relationship type
    first_pred = predictions[0]
    last_pred = predictions[-1]
    
    if first_pred > last_pred * 1.1:
        relationship = "Negative"
        impact_text = f"As {feature_name} increases, CPI decreases"
    elif last_pred > first_pred * 1.1:
        relationship = "Positive"
        impact_text = f"As {feature_name} increases, CPI increases"
    else:
        relationship = "Neutral"
        impact_text = f"{feature_name} has limited impact on CPI"
    
    # Add annotation for relationship
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"Relationship: {relationship}<br>{impact_text}",
        showarrow=False,
        font=dict(size=12, color=COLOR_SYSTEM['PRIMARY']['CONTRAST']),
        bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
        bordercolor=COLOR_SYSTEM['NEUTRAL']['DARK'],
        borderwidth=1,
        borderpad=4,
        align="left"
    )
    
    return fig

def create_model_comparison_chart(model_scores: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    Create a comprehensive model comparison visualization.
    
    Args:
        model_scores: Dictionary of model names to score metrics
        
    Returns:
        Plotly figure with model comparison
    """
    # Create metrics to display
    metrics = ['R²', 'RMSE', 'MAE']
    models = list(model_scores.keys())
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for each metric
    for i, metric in enumerate(metrics):
        values = [scores.get(metric, 0) for model, scores in model_scores.items()]
        
        # For RMSE and MAE, smaller is better, so invert for visualization consistency
        if metric in ['RMSE', 'MAE']:
            # Normalize to 0-1 range (inverted)
            max_val = max(values) if values else 1
            normalized = [1 - (val / max_val) if max_val > 0 else 0 for val in values]
            display_metric = f"{metric} (Lower is Better)"
        else:
            normalized = values
            display_metric = metric
        
        # Create text values for display
        text_values = [f"{values[j]:.4f}" for j in range(len(values))]
        
        # Add bar trace
        fig.add_trace(go.Bar(
            x=models,
            y=normalized,
            name=display_metric,
            text=text_values,
            textposition='auto',
            marker_color=COLOR_SYSTEM['CHARTS'][f'SERIES{i+1}' if i < 7 else 'SERIES1'],
            hovertemplate="Model: %{x}<br>" + f"{metric}: " + "%{text}<br><extra></extra>"
        ))
    
    # Format chart
    fig = format_chart_for_dark_mode(fig, "Model Comparison")
    
    fig.update_layout(
        barmode='group',
        xaxis_title="Model",
        yaxis_title="Score (normalized)",
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