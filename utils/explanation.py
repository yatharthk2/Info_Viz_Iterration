"""
Explanation utilities for the CPI Analysis & Prediction Dashboard.
Provides functions for explaining ML models and predictions.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explain_feature_importance(feature_importance_df: pd.DataFrame, num_features: int = 10) -> go.Figure:
    """
    Create an enhanced feature importance visualization.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame with Feature and Importance columns
        num_features (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly figure with feature importance visualization
    """
    try:
        if feature_importance_df.empty:
            return None
            
        # Sort and get top features
        df = feature_importance_df.sort_values('Importance', ascending=False).head(num_features)
        
        # Normalize importance for better visualization
        max_imp = df['Importance'].max()
        df['Normalized'] = df['Importance'] / max_imp
        
        # Create color gradient based on importance
        df['Color'] = df['Normalized'].apply(lambda x: f'rgba(78, 121, 167, {max(0.3, x)})')
        
        # Create the figure
        fig = go.Figure()
        
        # Add horizontal bars
        fig.add_trace(go.Bar(
            y=df['Feature'],
            x=df['Importance'],
            orientation='h',
            marker=dict(
                color=df['Color'],
                line=dict(color='rgba(255, 255, 255, 0.5)', width=1)
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        # Add markers to show relative scale
        fig.add_trace(go.Scatter(
            y=df['Feature'],
            x=df['Importance'],
            mode='markers',
            marker=dict(
                symbol='circle',
                size=12,
                color='rgba(255, 255, 255, 0.9)',
                line=dict(color='rgba(0, 0, 0, 0.5)', width=1)
            ),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Calculate dynamic padding based on the longest feature name
        max_len = df['Feature'].str.len().max()
        left_margin = min(150, max(80, max_len * 7))  # Adjust based on longest name
        
        # Update layout for better readability in dark mode
        fig.update_layout(
            title={
                'text': 'Feature Importance Analysis',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Relative Importance',
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=left_margin, r=30, t=80, b=30),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                categoryorder='total ascending',
                gridcolor='rgba(255,255,255,0.1)'
            ),
            font=dict(color='white'),
            height=400,
            bargap=0.15,
        )
        
        # Add annotations to explain what the importance means
        fig.add_annotation(
            text="Higher importance = stronger influence on predicted CPI",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,0.7)")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating feature importance visualization: {e}", exc_info=True)
        return None

def explain_prediction_factors(prediction: float, input_data: Dict[str, Any], 
                              top_factors: List[Tuple[str, float]]) -> str:
    """
    Generate a user-friendly explanation of what factors influenced the prediction.
    
    Args:
        prediction (float): The predicted CPI value
        input_data (Dict[str, Any]): The input data used for prediction
        top_factors (List[Tuple[str, float]]): List of (feature, importance) tuples
        
    Returns:
        str: Markdown-formatted explanation
    """
    explanation = f"""
    ### Key Factors Influencing This Prediction
    
    The model predicts a CPI of **${prediction:.2f}** based primarily on these key factors:
    
    """
    
    # Add bullet points for each top factor with interpretation
    for factor, importance in top_factors:
        if factor == "IR" or factor == "IR_Squared":
            explanation += f"- **Incidence Rate ({input_data.get('IR', 'N/A')}%)**: "
            if input_data.get('IR', 50) < 30:
                explanation += "Your low IR significantly increases the CPI as it's harder to find qualified respondents.\n"
            else:
                explanation += "Your relatively high IR helps keep the CPI in a moderate range.\n"
        
        elif factor == "LOI" or factor == "LOI_Squared":
            explanation += f"- **Length of Interview ({input_data.get('LOI', 'N/A')} min)**: "
            if input_data.get('LOI', 10) > 15:
                explanation += "The longer interview duration is increasing your CPI.\n"
            else:
                explanation += "The shorter interview duration is helping keep your CPI lower.\n"
        
        elif factor == "Completes" or factor == "Log_Completes":
            explanation += f"- **Sample Size ({input_data.get('Completes', 'N/A')})**: "
            if input_data.get('Completes', 200) > 500:
                explanation += "Your larger sample size is helping reduce the per-unit CPI through economies of scale.\n"
            else:
                explanation += "Your moderate sample size has a neutral effect on CPI.\n"
        
        elif "Ratio" in factor or "Product" in factor:
            explanation += f"- **Combined Factors**: The interaction between IR, LOI, and sample size is also influencing your price point.\n"
    
    # Add general interpretation
    explanation += """
    ### How to Interpret This
    
    - **Lower IR values** generally lead to higher CPIs due to the increased difficulty in finding qualified respondents
    - **Longer surveys (higher LOI)** typically require higher compensation, increasing the CPI
    - **Larger sample sizes** can reduce per-unit costs through economies of scale
    
    The model considers how these factors interact with each other to provide a comprehensive prediction.
    """
    
    return explanation

def create_prediction_comparison_with_explanation(predictions: Dict[str, float], 
                                                won_avg: float, lost_avg: float) -> go.Figure:
    """
    Create an enhanced prediction comparison chart with explanatory elements.
    
    Args:
        predictions (Dict[str, float]): Dictionary of model predictions
        won_avg (float): Average CPI for won bids
        lost_avg (float): Average CPI for lost bids
        
    Returns:
        go.Figure: Plotly figure with prediction comparison visualization
    """
    try:
        # Calculate average prediction
        avg_prediction = sum(predictions.values()) / len(predictions)
        
        # Create data structure for visualization
        models = list(predictions.keys())
        cpi_values = list(predictions.values())
        
        # Create a color scale with opacity based on how far from won_avg
        colors = []
        for val in cpi_values:
            # Calculate normalized distance to won_avg (0 = at won_avg, 1 = at or beyond lost_avg)
            if lost_avg > won_avg:
                norm_dist = min(1, max(0, (val - won_avg) / (lost_avg - won_avg)))
                # Color gradient from green to orange to red
                if norm_dist < 0.5:
                    color = f'rgba(82, 188, 163, {0.7 + norm_dist * 0.3})'  # Green to yellow-green
                else:
                    color = f'rgba(225, 87, 89, {0.7 + (norm_dist - 0.5) * 0.3})'  # Yellow-green to red
            else:
                # Fallback if lost_avg <= won_avg (unexpected case)
                color = 'rgba(82, 188, 163, 0.8)'  # Default green
            colors.append(color)
        
        # Create the figure
        fig = go.Figure()
        
        # Add bar chart for predictions
        fig.add_trace(go.Bar(
            x=models,
            y=cpi_values,
            marker_color=colors,
            text=[f"${val:.2f}" for val in cpi_values],
            textposition='auto',
            hovertemplate='<b>%{x} Model</b><br>Predicted CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Add reference lines for won and lost averages
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(models) - 0.5,
            y0=won_avg,
            y1=won_avg,
            line=dict(
                color="rgba(82, 188, 163, 0.8)",
                width=2,
                dash="solid"
            )
        )
        
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(models) - 0.5,
            y0=lost_avg,
            y1=lost_avg,
            line=dict(
                color="rgba(225, 87, 89, 0.8)",
                width=2,
                dash="solid"
            )
        )
        
        # Add average prediction line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(models) - 0.5,
            y0=avg_prediction,
            y1=avg_prediction,
            line=dict(
                color="rgba(255, 255, 255, 0.8)",
                width=3,
                dash="dash"
            )
        )
        
        # Add annotations for reference lines
        fig.add_annotation(
            x=len(models) - 0.5,
            y=won_avg,
            text=f"Won Avg: ${won_avg:.2f}",
            showarrow=False,
            font=dict(color="rgba(82, 188, 163, 1)"),
            xanchor="right",
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="rgba(82, 188, 163, 0.8)",
            borderwidth=1,
            borderpad=4,
            xshift=10
        )
        
        fig.add_annotation(
            x=len(models) - 0.5,
            y=lost_avg,
            text=f"Lost Avg: ${lost_avg:.2f}",
            showarrow=False,
            font=dict(color="rgba(225, 87, 89, 1)"),
            xanchor="right",
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor="rgba(225, 87, 89, 0.8)",
            borderwidth=1,
            borderpad=4,
            xshift=10
        )
        
        fig.add_annotation(
            x=len(models) - 0.5,
            y=avg_prediction,
            text=f"Avg Prediction: ${avg_prediction:.2f}",
            showarrow=False,
            font=dict(color="rgba(255, 255, 255, 1)"),
            xanchor="right",
            bgcolor="rgba(0, 0, 0, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.8)",
            borderwidth=1,
            borderpad=4,
            xshift=10
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI Prediction Comparison',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title={
                'text': 'Predicted CPI ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=40, r=40, t=80, b=80),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            ),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)'
            ),
            font=dict(color='white'),
            height=500,
            bargap=0.2,
        )
        
        # Add explanation for interpretation
        fig.add_annotation(
            text="💡 Predictions closer to the Won Average line have higher win probability",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=14, color="rgba(255,255,255,0.8)")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating prediction comparison visualization: {e}", exc_info=True)
        return None

def explain_win_probability(win_prob: float, similar_projects_count: int) -> str:
    """
    Generate an explanation for the win probability estimate.
    
    Args:
        win_prob (float): The estimated win probability percentage
        similar_projects_count (int): Number of similar projects used for estimation
        
    Returns:
        str: Markdown-formatted explanation
    """
    explanation = f"""
    ### Win Probability Explained
    
    The estimated win probability of **{win_prob:.1f}%** is based on analysis of {similar_projects_count} similar historical projects.
    
    How this is calculated:
    
    1. We identify projects with similar characteristics (IR, LOI, and sample size)
    2. We analyze what percentage of these similar projects were won at various price points
    3. We calculate where your predicted price falls within this distribution
    
    """
    
    # Add interpretation based on win probability
    if win_prob >= 80:
        explanation += """
        **Interpretation**: This is a **very high** win probability, suggesting your pricing is highly competitive.
        While you're likely to win, consider whether slightly higher pricing might still maintain a good win rate
        while improving profitability.
        """
    elif win_prob >= 60:
        explanation += """
        **Interpretation**: This is a **strong** win probability, suggesting your pricing is competitive
        while maintaining good profitability. This represents a good balance between winning bids and
        maximizing revenue.
        """
    elif win_prob >= 40:
        explanation += """
        **Interpretation**: This is a **moderate** win probability, suggesting your pricing is in a balanced
        range. You may want to adjust based on how important winning this specific bid is versus maintaining
        profitability targets.
        """
    elif win_prob >= 20:
        explanation += """
        **Interpretation**: This is a **below average** win probability, suggesting your pricing may be too
        high for this market segment. Consider lowering your price if winning this bid is important.
        """
    else:
        explanation += """
        **Interpretation**: This is a **low** win probability, suggesting your pricing is significantly higher
        than what typically wins in this market segment. A substantial price reduction may be needed to become
        competitive.
        """
    
    return explanation

def create_price_sensitivity_curve(price_points: List[float], 
                                 win_probabilities: List[float], 
                                 predicted_price: float) -> go.Figure:
    """
    Create a price sensitivity curve showing how win probability changes with different prices.
    
    Args:
        price_points (List[float]): List of CPI price points
        win_probabilities (List[float]): Corresponding win probabilities
        predicted_price (float): The current predicted price
        
    Returns:
        go.Figure: Plotly figure with price sensitivity curve
    """
    try:
        # Create the figure
        fig = go.Figure()
        
        # Add line chart
        fig.add_trace(go.Scatter(
            x=price_points,
            y=win_probabilities,
            mode='lines+markers',
            line=dict(color='rgba(78, 121, 167, 0.8)', width=3),
            marker=dict(size=8, color='rgba(78, 121, 167, 0.9)'),
            hovertemplate='<b>Price: $%{x:.2f}</b><br>Win Probability: %{y:.1f}%<extra></extra>'
        ))
        
        # Add point for current prediction
        current_prob_idx = min(range(len(price_points)), key=lambda i: abs(price_points[i] - predicted_price))
        current_prob = win_probabilities[current_prob_idx]
        
        fig.add_trace(go.Scatter(
            x=[predicted_price],
            y=[current_prob],
            mode='markers',
            marker=dict(
                size=14,
                color='rgba(255, 255, 255, 1)',
                line=dict(color='rgba(0, 0, 0, 1)', width=2)
            ),
            hovertemplate='<b>Your Prediction: $%{x:.2f}</b><br>Win Probability: %{y:.1f}%<extra></extra>',
            name='Your Prediction'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Price Sensitivity Analysis',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'CPI Price Point ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': 'Win Probability (%)',
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=40, r=40, t=80, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                ticksuffix='%',
                range=[0, 100]
            ),
            font=dict(color='white'),
            height=400,
            showlegend=False
        )
        
        # Add annotations to explain what the curve means
        fig.add_annotation(
            text="This curve shows how win probability changes at different price points",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,0.7)")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating price sensitivity curve: {e}", exc_info=True)
        return None
