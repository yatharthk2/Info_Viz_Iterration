"""
Visualization utilities for the CPI Analysis & Prediction Dashboard.
Provides functions for creating charts and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define color schemes for dark theme
DARK_THEME_COLORS = {
    'won': '#52bca3',  # Teal/green for won bids
    'lost': '#e15759',  # Red for lost bids
    'neutral': '#4e79a7',  # Blue for neutral
    'highlight': '#f28e2b',  # Orange for highlights
    'gradient': ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc949'],
    'sequential': ['#f0f0f0', '#bdbdbd', '#767676', '#404040', '#1a1a1a']
}

def format_chart_for_dark_mode(fig: go.Figure) -> go.Figure:
    """
    Apply common dark mode formatting to a Plotly figure.
    
    Args:
        fig (go.Figure): Plotly figure to format
        
    Returns:
        go.Figure: Formatted figure
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        margin=dict(l=40, r=40, t=80, b=40),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        legend=dict(
            font=dict(color='rgba(255,255,255,0.8)'),
            bgcolor='rgba(0,0,0,0.2)',
            bordercolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_type_distribution_chart(combined_data: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing distribution of won vs lost bids.
    
    Args:
        combined_data (pd.DataFrame): DataFrame with both won and lost bids
        
    Returns:
        go.Figure: Plotly figure with pie chart
    """
    try:
        # Count won and lost bids
        type_counts = combined_data['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        # Calculate percentages
        total = type_counts['Count'].sum()
        type_counts['Percentage'] = (type_counts['Count'] / total * 100).round(1)
        type_counts['Label'] = type_counts.apply(
            lambda x: f"{x['Type']}: {x['Count']} ({x['Percentage']}%)", axis=1
        )
        
        # Create color mapping
        color_map = {'Won': DARK_THEME_COLORS['won'], 'Lost': DARK_THEME_COLORS['lost']}
        
        # Create pie chart
        fig = px.pie(
            type_counts, 
            values='Count', 
            names='Type',
            color='Type',
            color_discrete_map=color_map,
            title='Distribution of Won vs Lost Bids',
            labels={'Count': 'Number of Bids'},
            hole=0.4
        )
        
        # Update text info
        fig.update_traces(
            textinfo='label+percent',
            textfont_size=14,
            textfont_color='white',
            marker=dict(line=dict(color='#000000', width=1.5))
        )
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Add custom styling
        fig.update_layout(
            title={
                'text': 'Distribution of Won vs Lost Bids',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating type distribution chart: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced boxplot showing CPI distribution for won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        
    Returns:
        go.Figure: Plotly figure with boxplot
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add box plots with added styling for dark mode
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won Bids',
            marker_color=DARK_THEME_COLORS['won'],
            boxmean=True,
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                opacity=0.7,
                size=6,
                line=dict(width=1, color='rgba(0,0,0,0.3)')
            ),
            hoverinfo='y',
            hovertemplate='<b>CPI: $%{y:.2f}</b><extra></extra>'
        ))
        
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost Bids',
            marker_color=DARK_THEME_COLORS['lost'],
            boxmean=True,
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                opacity=0.7,
                size=6,
                line=dict(width=1, color='rgba(0,0,0,0.3)')
            ),
            hoverinfo='y',
            hovertemplate='<b>CPI: $%{y:.2f}</b><extra></extra>'
        ))
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Update layout with custom styling
        fig.update_layout(
            title={
                'text': 'CPI Distribution: Won vs Lost Bids',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title={
                'text': 'CPI ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis=dict(
                tickprefix='$',
                showgrid=True
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            boxmode='group',
            height=400
        )
        
        # Add annotations to show mean values
        won_mean = won_data['CPI'].mean()
        lost_mean = lost_data['CPI'].mean()
        
        fig.add_annotation(
            x=0,
            y=won_mean,
            text=f"Mean: ${won_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=DARK_THEME_COLORS['won'],
            font=dict(color=DARK_THEME_COLORS['won']),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=DARK_THEME_COLORS['won'],
            borderwidth=1,
            borderpad=4,
            xshift=40
        )
        
        fig.add_annotation(
            x=1,
            y=lost_mean,
            text=f"Mean: ${lost_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=DARK_THEME_COLORS['lost'],
            font=dict(color=DARK_THEME_COLORS['lost']),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=DARK_THEME_COLORS['lost'],
            borderwidth=1,
            borderpad=4,
            xshift=-40
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI distribution boxplot: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced scatter plot showing CPI vs IR relationship.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        
    Returns:
        go.Figure: Plotly figure with scatter plot
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for won bids with styled markers
        fig.add_trace(go.Scatter(
            x=won_data['IR'],
            y=won_data['CPI'],
            mode='markers',
            name='Won Bids',
            marker=dict(
                color=DARK_THEME_COLORS['won'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                symbol='circle'
            ),
            hovertemplate='<b>Won Bid</b><br>IR: %{x}%<br>CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Add scatter plot for lost bids with styled markers
        fig.add_trace(go.Scatter(
            x=lost_data['IR'],
            y=lost_data['CPI'],
            mode='markers',
            name='Lost Bids',
            marker=dict(
                color=DARK_THEME_COLORS['lost'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                symbol='circle'
            ),
            hovertemplate='<b>Lost Bid</b><br>IR: %{x}%<br>CPI: $%{y:.2f}<extra></extra>'
        ))
        
        # Add trend lines for won bids
        x_range = np.linspace(won_data['IR'].min(), won_data['IR'].max(), 100)
        
        # Simple trend line calculation - polynomial fit for won data
        if len(won_data) >= 3:  # Need at least 3 points for quadratic fit
            won_z = np.polyfit(won_data['IR'], won_data['CPI'], 2)
            won_p = np.poly1d(won_z)
            won_trend_y = won_p(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=won_trend_y,
                mode='lines',
                name='Won Trend',
                line=dict(
                    color=DARK_THEME_COLORS['won'],
                    width=3,
                    dash='solid'
                ),
                hoverinfo='skip'
            ))
        
        # Simple trend line calculation - polynomial fit for lost data
        if len(lost_data) >= 3:  # Need at least 3 points for quadratic fit
            lost_z = np.polyfit(lost_data['IR'], lost_data['CPI'], 2)
            lost_p = np.poly1d(lost_z)
            lost_trend_y = lost_p(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=lost_trend_y,
                mode='lines',
                name='Lost Trend',
                line=dict(
                    color=DARK_THEME_COLORS['lost'],
                    width=3,
                    dash='solid'
                ),
                hoverinfo='skip'
            ))
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Update layout with custom styling
        fig.update_layout(
            title={
                'text': 'CPI vs Incidence Rate (IR)',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Incidence Rate (%)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': 'CPI ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis=dict(
                tickprefix='$',
                showgrid=True
            ),
            xaxis=dict(
                ticksuffix='%',
                showgrid=True
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            height=500
        )
        
        # Add explanation annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            text="As IR increases, CPI typically decreases",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.7)", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI vs IR scatter plot: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced CPI efficiency visualization.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        
    Returns:
        go.Figure: Plotly figure with efficiency chart
    """
    try:
        # Create a combined efficiency metric
        # For example: IR * (1/LOI) * log(Completes) - higher values should correlate with lower CPIs
        won_data_copy = won_data.copy()
        lost_data_copy = lost_data.copy()
        
        # Ensure we don't have zeros that would cause division issues
        won_data_copy['LOI'] = won_data_copy['LOI'].replace(0, 0.1)
        lost_data_copy['LOI'] = lost_data_copy['LOI'].replace(0, 0.1)
        
        # Calculate efficiency metric
        won_data_copy['Efficiency'] = won_data_copy['IR'] * (1/won_data_copy['LOI']) * np.log1p(won_data_copy['Completes'])
        lost_data_copy['Efficiency'] = lost_data_copy['IR'] * (1/lost_data_copy['LOI']) * np.log1p(lost_data_copy['Completes'])
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for won bids
        fig.add_trace(go.Scatter(
            x=won_data_copy['Efficiency'],
            y=won_data_copy['CPI'],
            mode='markers',
            name='Won Bids',
            marker=dict(
                color=DARK_THEME_COLORS['won'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                symbol='circle'
            ),
            hovertemplate='<b>Won Bid</b><br>Efficiency Score: %{x:.2f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]}%<br>LOI: %{customdata[1]} min<extra></extra>',
            customdata=np.column_stack((won_data_copy['IR'], won_data_copy['LOI']))
        ))
        
        # Add scatter plot for lost bids
        fig.add_trace(go.Scatter(
            x=lost_data_copy['Efficiency'],
            y=lost_data_copy['CPI'],
            mode='markers',
            name='Lost Bids',
            marker=dict(
                color=DARK_THEME_COLORS['lost'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)'),
                symbol='circle'
            ),
            hovertemplate='<b>Lost Bid</b><br>Efficiency Score: %{x:.2f}<br>CPI: $%{y:.2f}<br>IR: %{customdata[0]}%<br>LOI: %{customdata[1]} min<extra></extra>',
            customdata=np.column_stack((lost_data_copy['IR'], lost_data_copy['LOI']))
        ))
        
        # Add trend lines
        won_x_range = np.linspace(won_data_copy['Efficiency'].min(), won_data_copy['Efficiency'].max(), 100)
        lost_x_range = np.linspace(lost_data_copy['Efficiency'].min(), lost_data_copy['Efficiency'].max(), 100)
        
        # Simple trend line calculation for won data
        if len(won_data_copy) >= 2:
            won_z = np.polyfit(won_data_copy['Efficiency'], won_data_copy['CPI'], 1)
            won_p = np.poly1d(won_z)
            won_trend_y = won_p(won_x_range)
            
            fig.add_trace(go.Scatter(
                x=won_x_range,
                y=won_trend_y,
                mode='lines',
                name='Won Trend',
                line=dict(
                    color=DARK_THEME_COLORS['won'],
                    width=3,
                    dash='solid'
                ),
                hoverinfo='skip'
            ))
        
        # Simple trend line calculation for lost data
        if len(lost_data_copy) >= 2:
            lost_z = np.polyfit(lost_data_copy['Efficiency'], lost_data_copy['CPI'], 1)
            lost_p = np.poly1d(lost_z)
            lost_trend_y = lost_p(lost_x_range)
            
            fig.add_trace(go.Scatter(
                x=lost_x_range,
                y=lost_trend_y,
                mode='lines',
                name='Lost Trend',
                line=dict(
                    color=DARK_THEME_COLORS['lost'],
                    width=3,
                    dash='solid'
                ),
                hoverinfo='skip'
            ))
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI vs Project Efficiency Score',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Efficiency Score (IR÷LOI×log(Completes))',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': 'CPI ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis=dict(
                tickprefix='$',
                showgrid=True
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5
            ),
            height=500
        )
        
        # Add explanation annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            text="Higher efficiency score = better project parameters (higher IR, lower LOI, larger sample)",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.7)", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI efficiency chart: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()

def create_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced feature importance visualization.
    
    Args:
        feature_importance (pd.DataFrame): DataFrame with Feature and Importance columns
        
    Returns:
        go.Figure: Plotly figure with feature importance visualization
    """
    try:
        if feature_importance.empty:
            return go.Figure()
            
        # Sort and get top features
        df = feature_importance.sort_values('Importance', ascending=False).head(10)
        
        # Create a color gradient based on importance rank
        n_features = len(df)
        colors = [f'rgba(78, 121, 167, {max(0.3, 1 - (i / n_features))})' for i in range(n_features)]
        
        # Create the figure
        fig = go.Figure()
        
        # Add horizontal bars
        fig.add_trace(go.Bar(
            y=df['Feature'],
            x=df['Importance'],
            orientation='h',
            marker_color=colors,
            text=df['Importance'].apply(lambda x: f'{x:.4f}'),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Feature Importance Analysis',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Relative Importance',
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=150, r=30, t=80, b=30),
            height=400
        )
        
        # Update axes settings
        fig.update_yaxes(
            categoryorder='total ascending'
        )
        
        # Add annotation explaining feature importance
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            text="Higher values indicate stronger influence on predicted CPI",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.7)", size=12),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=4
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()

def create_prediction_comparison_chart(predictions: Dict[str, float], 
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
        
        # Create color gradient based on position relative to won/lost average
        colors = []
        for val in cpi_values:
            if val <= won_avg:
                # Below won average - green-blue gradient
                norm_val = max(0, val / won_avg)
                color = f'rgba(82, 188, 163, {0.6 + 0.4 * norm_val})'
            elif val <= lost_avg:
                # Between won and lost - orange gradient
                norm_val = (val - won_avg) / (lost_avg - won_avg)
                color = f'rgba(242, 142, 43, {0.6 + 0.4 * norm_val})'
            else:
                # Above lost average - red gradient
                color = 'rgba(225, 87, 89, 0.9)'
            colors.append(color)
        
        # Create figure
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
                color=DARK_THEME_COLORS['won'],
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
                color=DARK_THEME_COLORS['lost'],
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
            font=dict(color=DARK_THEME_COLORS['won']),
            xanchor="right",
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor=DARK_THEME_COLORS['won'],
            borderwidth=1,
            borderpad=4,
            xshift=10
        )
        
        fig.add_annotation(
            x=len(models) - 0.5,
            y=lost_avg,
            text=f"Lost Avg: ${lost_avg:.2f}",
            showarrow=False,
            font=dict(color=DARK_THEME_COLORS['lost']),
            xanchor="right",
            bgcolor="rgba(0, 0, 0, 0.5)",
            bordercolor=DARK_THEME_COLORS['lost'],
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
        
        # Format for dark mode
        fig = format_chart_for_dark_mode(fig)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI Prediction Comparison',
                'font': {'size': 18, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title={
                'text': 'Predicted CPI ($)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis=dict(
                tickprefix='$'
            ),
            height=400
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating prediction comparison chart: {e}", exc_info=True)
        # Return an empty figure
        return go.Figure()
