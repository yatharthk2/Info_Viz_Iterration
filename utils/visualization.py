"""
Visualization utilities for the CPI Analysis & Prediction Dashboard.
Provides enhanced visualizations optimized for dark mode and high contrast.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dark theme colors consistently applied across all visualizations
DARK_THEME_COLORS = {
    "background": "#111111",
    "secondary_background": "#1E2130",
    "primary": "#4e79a7",
    "success": "#52BC9F",
    "warning": "#F6C85F",
    "error": "#E15759",
    "text": "#FFFFFF",
    "secondary_text": "#BBBBBB",
    "highlight": "#7EB3FF",
    "muted": "#555555",
    "won": "#52BC9F",
    "lost": "#E15759",
    "neutral": "#7EB3FF",
    "grid": "rgba(255, 255, 255, 0.1)",
}

# Color scales for consistent visualization
COLOR_SCALES = {
    "diverging": [[0, "#E15759"], [0.5, "#F6C85F"], [1, "#52BC9F"]],
    "sequential": [[0, "#4e79a7"], [0.5, "#7EB3FF"], [1, "#52BC9F"]],
    "sequential_red": [[0, "#E15759"], [0.5, "#F08B6E"], [1, "#F6C85F"]],
}

def set_plotly_theme() -> None:
    """
    Set global Plotly theme settings for consistent styling across all visualizations.
    Optimized for dark background with high-contrast elements.
    """
    import plotly.io as pio
    
    pio.templates.default = "plotly_dark"
    
    # Customize the dark template further
    template = pio.templates["plotly_dark"]
    template.layout.update(
        paper_bgcolor=DARK_THEME_COLORS["background"],
        plot_bgcolor=DARK_THEME_COLORS["background"],
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color=DARK_THEME_COLORS["text"]
        ),
        title=dict(
            font=dict(
                size=20,
                color=DARK_THEME_COLORS["text"]
            )
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.2)",
            bordercolor=DARK_THEME_COLORS["muted"],
            borderwidth=1,
            font=dict(
                size=12,
                color=DARK_THEME_COLORS["text"]
            )
        ),
        xaxis=dict(
            gridcolor=DARK_THEME_COLORS["grid"],
            zerolinecolor=DARK_THEME_COLORS["grid"]
        ),
        yaxis=dict(
            gridcolor=DARK_THEME_COLORS["grid"],
            zerolinecolor=DARK_THEME_COLORS["grid"]
        ),
        margin=dict(t=60, l=40, r=40, b=60),
        autosize=True,
        hovermode="closest"
    )

def create_type_distribution_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of won vs lost bids.
    
    Args:
        data (pd.DataFrame): DataFrame with Type column
        
    Returns:
        go.Figure: Plotly figure with type distribution visualization
    """
    try:
        # Count the number of won and lost bids
        type_counts = data['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        # Calculate percentages
        total = type_counts['Count'].sum()
        type_counts['Percentage'] = type_counts['Count'] / total * 100
        
        # Create custom text
        type_counts['Text'] = type_counts.apply(
            lambda x: f"{x['Type']}: {x['Count']} ({x['Percentage']:.1f}%)", axis=1)
        
        # Create color map based on Type
        colors = [DARK_THEME_COLORS['won'] if t == 'Won' else DARK_THEME_COLORS['lost'] 
                 for t in type_counts['Type']]
        
        # Create pie chart with Plotly
        fig = go.Figure(data=[go.Pie(
            labels=type_counts['Type'],
            values=type_counts['Count'],
            text=type_counts['Text'],
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
            textinfo='text',
            marker=dict(
                colors=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            hole=0.4,
        )])
        
        # Add title and styling
        fig.update_layout(
            title={
                'text': 'Distribution of Won vs Lost Bids',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            legend=dict(
                font=dict(color='white', size=12),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            annotations=[dict(
                text='Total Bids:<br><b>' + str(total) + '</b>',
                x=0.5, y=0.5,
                font=dict(size=14, color='white'),
                showarrow=False
            )]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating type distribution chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a boxplot showing the distribution of CPI values for won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with CPI distribution visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add box plot for won bids
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won Bids',
            marker_color=DARK_THEME_COLORS['won'],
            boxmean=True,
            boxpoints='outliers',
            jitter=0.3,
            whiskerwidth=0.2,
            marker=dict(
                size=4,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            line=dict(width=2),
            hoverinfo='y',
            hovertemplate="<b>Won Bid</b><br>CPI: $%{y:.2f}<extra></extra>"
        ))
        
        # Add box plot for lost bids
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost Bids',
            marker_color=DARK_THEME_COLORS['lost'],
            boxmean=True,
            boxpoints='outliers',
            jitter=0.3,
            whiskerwidth=0.2,
            marker=dict(
                size=4,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            line=dict(width=2),
            hoverinfo='y',
            hovertemplate="<b>Lost Bid</b><br>CPI: $%{y:.2f}<extra></extra>"
        ))
        
        # Add mean lines and annotations
        won_mean = won_data['CPI'].mean()
        lost_mean = lost_data['CPI'].mean()
        diff_pct = ((lost_mean - won_mean) / won_mean * 100) if won_mean > 0 else 0
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI Distribution: Won vs Lost Bids',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis=dict(
                title='CPI ($)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            legend=dict(
                font=dict(color='white', size=12),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=40, t=80, b=50),
            height=500,
            annotations=[
                dict(
                    x=0, y=won_mean,
                    xref='x', yref='y',
                    text=f"Mean: ${won_mean:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=DARK_THEME_COLORS['won'],
                    ax=-80, ay=0,
                    font=dict(color=DARK_THEME_COLORS['won'], size=12),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=4
                ),
                dict(
                    x=1, y=lost_mean,
                    xref='x', yref='y',
                    text=f"Mean: ${lost_mean:.2f}<br>+{diff_pct:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=DARK_THEME_COLORS['lost'],
                    ax=80, ay=0,
                    font=dict(color=DARK_THEME_COLORS['lost'], size=12),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=4
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI distribution boxplot: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of CPI vs IR for won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with scatter plot visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for won bids
        fig.add_trace(go.Scatter(
            x=won_data['IR'],
            y=won_data['CPI'],
            mode='markers',
            name='Won Bids',
            marker=dict(
                color=DARK_THEME_COLORS['won'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            hovertemplate="<b>Won Bid</b><br>IR: %{x}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]} min<br>Completes: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((won_data['LOI'], won_data['Completes']))
        ))
        
        # Add scatter plot for lost bids
        fig.add_trace(go.Scatter(
            x=lost_data['IR'],
            y=lost_data['CPI'],
            mode='markers',
            name='Lost Bids',
            marker=dict(
                color=DARK_THEME_COLORS['lost'],
                size=10,
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            hovertemplate="<b>Lost Bid</b><br>IR: %{x}%<br>CPI: $%{y:.2f}<br>LOI: %{customdata[0]} min<br>Completes: %{customdata[1]}<extra></extra>",
            customdata=np.column_stack((lost_data['LOI'], lost_data['Completes']))
        ))
        
        # Add trendlines
        if len(won_data) >= 2:
            # Won bids trendline
            z = np.polyfit(won_data['IR'], won_data['CPI'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(won_data['IR'].min(), won_data['IR'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Won Trend',
                line=dict(color=DARK_THEME_COLORS['won'], width=2, dash='dash'),
                hoverinfo='skip'
            ))
        
        if len(lost_data) >= 2:
            # Lost bids trendline
            z = np.polyfit(lost_data['IR'], lost_data['CPI'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(lost_data['IR'].min(), lost_data['IR'].max(), 100)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=p(x_range),
                mode='lines',
                name='Lost Trend',
                line=dict(color=DARK_THEME_COLORS['lost'], width=2, dash='dash'),
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI vs Incidence Rate (IR)',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Incidence Rate (%)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                ticksuffix='%',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title='CPI ($)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            legend=dict(
                font=dict(color='white', size=12),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=40, t=80, b=50),
            height=500,
            annotations=[
                dict(
                    x=0.5, y=1.12,
                    xref='paper', yref='paper',
                    text='Lower IR generally results in higher CPI as qualifying respondents become harder to find',
                    showarrow=False,
                    font=dict(color='rgba(255,255,255,0.7)', size=12),
                    bgcolor='rgba(0,0,0,0.3)',
                    borderpad=4,
                    borderwidth=1,
                    bordercolor='rgba(255,255,255,0.3)',
                    width=600,
                    align='center'
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI vs IR scatter plot: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing CPI versus a calculated efficiency metric.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with efficiency visualization
    """
    try:
        # Calculate efficiency metric (CPI per minute of LOI, adjusted by IR)
        # Higher value means less efficient
        won_data['Efficiency'] = won_data['CPI'] / (won_data['LOI'] * (won_data['IR'] / 100))
        lost_data['Efficiency'] = lost_data['CPI'] / (lost_data['LOI'] * (lost_data['IR'] / 100))
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for won bids with size based on Completes
        fig.add_trace(go.Scatter(
            x=won_data['LOI'],
            y=won_data['Efficiency'],
            mode='markers',
            name='Won Bids',
            marker=dict(
                color=DARK_THEME_COLORS['won'],
                size=won_data['Completes'] / won_data['Completes'].max() * 30 + 5,  # Scale the size
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            hovertemplate="<b>Won Bid</b><br>LOI: %{x} min<br>Efficiency Score: %{y:.2f}<br>IR: %{customdata[0]}%<br>CPI: $%{customdata[1]:.2f}<br>Completes: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((won_data['IR'], won_data['CPI'], won_data['Completes']))
        ))
        
        # Add scatter plot for lost bids with size based on Completes
        fig.add_trace(go.Scatter(
            x=lost_data['LOI'],
            y=lost_data['Efficiency'],
            mode='markers',
            name='Lost Bids',
            marker=dict(
                color=DARK_THEME_COLORS['lost'],
                size=lost_data['Completes'] / lost_data['Completes'].max() * 30 + 5,  # Scale the size
                opacity=0.7,
                line=dict(width=1, color='rgba(255,255,255,0.3)')
            ),
            hovertemplate="<b>Lost Bid</b><br>LOI: %{x} min<br>Efficiency Score: %{y:.2f}<br>IR: %{customdata[0]}%<br>CPI: $%{customdata[1]:.2f}<br>Completes: %{customdata[2]}<extra></extra>",
            customdata=np.column_stack((lost_data['IR'], lost_data['CPI'], lost_data['Completes']))
        ))
        
        # Add horizontal line for average won efficiency
        won_avg_efficiency = won_data['Efficiency'].mean()
        
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=won_avg_efficiency,
            y1=won_avg_efficiency,
            xref="paper",
            line=dict(
                color=DARK_THEME_COLORS['won'],
                width=2,
                dash="dash",
            )
        )
        
        # Add horizontal line for average lost efficiency
        lost_avg_efficiency = lost_data['Efficiency'].mean()
        
        fig.add_shape(
            type="line",
            x0=0,
            x1=1,
            y0=lost_avg_efficiency,
            y1=lost_avg_efficiency,
            xref="paper",
            line=dict(
                color=DARK_THEME_COLORS['lost'],
                width=2,
                dash="dash",
            )
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Pricing Efficiency by Length of Interview (LOI)',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis=dict(
                title='Length of Interview (minutes)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title='Efficiency Score (lower is better)',
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                titlefont=dict(color='white'),
                tickfont=dict(color='white')
            ),
            legend=dict(
                font=dict(color='white', size=12),
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=50, r=40, t=100, b=50),
            height=500,
            annotations=[
                dict(
                    x=0.5, y=1.12,
                    xref='paper', yref='paper',
                    text='Bubble size represents sample size (number of completes)',
                    showarrow=False,
                    font=dict(color='rgba(255,255,255,0.7)', size=12),
                    bgcolor='rgba(0,0,0,0.3)',
                    borderpad=4,
                    borderwidth=1,
                    bordercolor='rgba(255,255,255,0.3)',
                    width=600,
                    align='center'
                ),
                dict(
                    x=1, y=won_avg_efficiency,
                    xref='paper', yref='y',
                    text=f"Won Avg: {won_avg_efficiency:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=DARK_THEME_COLORS['won'],
                    ax=50, ay=0,
                    font=dict(color=DARK_THEME_COLORS['won'], size=12),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=4
                ),
                dict(
                    x=1, y=lost_avg_efficiency,
                    xref='paper', yref='y',
                    text=f"Lost Avg: {lost_avg_efficiency:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=DARK_THEME_COLORS['lost'],
                    ax=50, ay=0,
                    font=dict(color=DARK_THEME_COLORS['lost'], size=12),
                    bgcolor='rgba(0,0,0,0.5)',
                    borderpad=4
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI efficiency chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_cpi_comparison_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create an enhanced CPI comparison chart between won and lost bids with improved readability
    on dark backgrounds and accessibility features.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with CPI comparison visualization
    """
    try:
        # Calculate statistics
        won_mean = won_data['CPI'].mean()
        lost_mean = lost_data['CPI'].mean()
        
        won_median = won_data['CPI'].median()
        lost_median = lost_data['CPI'].median()
        
        # Create the figure with separate subplots for better control
        fig = go.Figure()
        
        # Add box plots for won and lost bids
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won Bids',
            boxmean=True,  # Show mean
            marker_color=DARK_THEME_COLORS['won'],
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hoverinfo='y',
            hovertemplate='CPI: $%{y:.2f}<extra>Won Bid</extra>'
        ))
        
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost Bids',
            boxmean=True,  # Show mean
            marker_color=DARK_THEME_COLORS['lost'],
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hoverinfo='y',
            hovertemplate='CPI: $%{y:.2f}<extra>Lost Bid</extra>'
        ))
        
        # Add additional visual enhancements for context
        
        # Mean value lines
        fig.add_shape(
            type="line",
            x0=-0.4, x1=0.4,
            y0=won_mean, y1=won_mean,
            line=dict(
                color="white",
                width=2,
                dash="dot",
            ),
            xref='x', yref='y'
        )
        
        fig.add_shape(
            type="line",
            x0=0.6, x1=1.4,
            y0=lost_mean, y1=lost_mean,
            line=dict(
                color="white",
                width=2,
                dash="dot",
            ),
            xref='x', yref='y'
        )
        
        # Add annotations for the statistics
        fig.add_annotation(
            x=0, 
            y=won_mean,
            text=f"Mean: ${won_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
            ax=-40,
            ay=-30,
            bgcolor=DARK_THEME_COLORS['won'],
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            font=dict(color="white", size=12)
        )
        
        fig.add_annotation(
            x=1, 
            y=lost_mean,
            text=f"Mean: ${lost_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
            ax=40,
            ay=-30,
            bgcolor=DARK_THEME_COLORS['lost'],
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            font=dict(color="white", size=12)
        )
        
        # Price gap annotation
        fig.add_annotation(
            x=0.5,
            y=(won_mean + lost_mean) / 2,
            text=f"Price Gap: ${lost_mean - won_mean:.2f} ({((lost_mean - won_mean) / won_mean * 100):.1f}%)",
            showarrow=False,
            font=dict(size=14, color=DARK_THEME_COLORS['highlight']),
            bordercolor=DARK_THEME_COLORS['highlight'],
            borderwidth=2,
            borderpad=4,
            bgcolor="rgba(0, 0, 0, 0.6)",
            opacity=0.8
        )
        
        # Update layout with enhanced styling for dark mode
        fig.update_layout(
            title={
                'text': "CPI Comparison: Won vs. Lost Bids",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 22, 'color': 'white'}
            },
            yaxis_title={
                'text': "Cost Per Interview (CPI) in $",
                'font': {'size': 16, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            ),
            font=dict(color='white'),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor='rgba(0,0,0,0.2)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1
            )
        )
        
        # Add an explanation annotation at the bottom
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            text=f"The average CPI for Won bids (${won_mean:.2f}) is {((lost_mean - won_mean) / won_mean * 100):.1f}% lower than Lost bids (${lost_mean:.2f})",
            showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,0.7)"),
            align="center"
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI comparison chart: {e}", exc_info=True)
        return go.Figure()

def create_feature_importance_chart(feature_importance_df: pd.DataFrame, num_features: int = 10) -> go.Figure:
    """
    Create an enhanced feature importance visualization with high contrast and informative elements.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame with Feature and Importance columns
        num_features (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly figure with feature importance visualization
    """
    try:
        if feature_importance_df.empty:
            # Return an empty figure with a message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=14, color="white")
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=400
            )
            return fig
            
        # Sort and get top features
        df = feature_importance_df.sort_values('Importance', ascending=False).head(num_features)
        
        # Normalize importance
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
        left_margin = min(150, max(80, max_len * 7))
        
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
        logger.error(f"Error creating feature importance chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig

def create_prediction_comparison_chart(predictions: Dict[str, float], 
                                     won_avg: float, lost_avg: float) -> go.Figure:
    """
    Create a prediction comparison chart with enhanced visual cues for optimal pricing.
    
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
        logger.error(f"Error creating prediction comparison chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_win_probability_chart(price_points: List[float], win_probabilities: List[float], 
                              predicted_price: float) -> go.Figure:
    """
    Create a win probability curve showing how win probability changes with price.
    
    Args:
        price_points (List[float]): List of price points
        win_probabilities (List[float]): Corresponding win probabilities
        predicted_price (float): The predicted price
        
    Returns:
        go.Figure: Plotly figure with win probability visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add the win probability curve
        fig.add_trace(go.Scatter(
            x=price_points,
            y=win_probabilities,
            mode='lines',
            line=dict(
                color=DARK_THEME_COLORS['primary'],
                width=3,
                shape='spline'
            ),
            hovertemplate='CPI: $%{x:.2f}<br>Win Probability: %{y:.1f}%<extra></extra>',
            name='Win Probability'
        ))
        
        # Add markers to highlight key points
        # Find the probability at the predicted price
        predicted_idx = 0
        for i, price in enumerate(price_points):
            if price >= predicted_price:
                predicted_idx = i
                break
        
        predicted_prob = win_probabilities[predicted_idx]
        
        # Add marker for predicted price
        fig.add_trace(go.Scatter(
            x=[predicted_price],
            y=[predicted_prob],
            mode='markers',
            marker=dict(
                size=14,
                color=DARK_THEME_COLORS['highlight'],
                line=dict(color='white', width=2),
                symbol='circle'
            ),
            hovertemplate='Predicted CPI: $%{x:.2f}<br>Win Probability: %{y:.1f}%<extra>Recommended</extra>',
            name='Predicted Price'
        ))
        
        # Find high probability point (>80%)
        high_prob_idx = next((i for i, prob in enumerate(win_probabilities) if prob >= 80), None)
        
        if high_prob_idx is not None:
            high_prob_price = price_points[high_prob_idx]
            fig.add_trace(go.Scatter(
                x=[high_prob_price],
                y=[win_probabilities[high_prob_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=DARK_THEME_COLORS['won'],
                    line=dict(color='white', width=1),
                    symbol='diamond'
                ),
                hovertemplate='CPI: $%{x:.2f}<br>Win Probability: %{y:.1f}%<extra>High Win Chance</extra>',
                name='High Win Probability'
            ))
        
        # Find optimal balance point (around 65-70% probability)
        balance_idx = next((i for i, prob in enumerate(win_probabilities) if prob <= 70), None)
        
        if balance_idx is not None:
            balance_price = price_points[balance_idx]
            fig.add_trace(go.Scatter(
                x=[balance_price],
                y=[win_probabilities[balance_idx]],
                mode='markers',
                marker=dict(
                    size=12,
                    color=DARK_THEME_COLORS['warning'],
                    line=dict(color='white', width=1),
                    symbol='square'
                ),
                hovertemplate='CPI: $%{x:.2f}<br>Win Probability: %{y:.1f}%<extra>Balanced</extra>',
                name='Balanced Price Point'
            ))
        
        # Shade the area under the curve
        fig.add_trace(go.Scatter(
            x=price_points,
            y=win_probabilities,
            fill='tozeroy',
            fillcolor='rgba(78, 121, 167, 0.3)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add vertical line at predicted price
        fig.add_shape(
            type="line",
            x0=predicted_price, x1=predicted_price,
            y0=0, y1=predicted_prob,
            line=dict(
                color=DARK_THEME_COLORS['highlight'],
                width=2,
                dash="dash"
            )
        )
        
        # Add annotation for the predicted price
        fig.add_annotation(
            x=predicted_price,
            y=predicted_prob + 5,  # Slightly above the point
            text=f"Predicted<br>${predicted_price:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=DARK_THEME_COLORS['highlight'],
            ax=0,
            ay=-30,
            font=dict(size=12, color=DARK_THEME_COLORS['highlight']),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=DARK_THEME_COLORS['highlight'],
            borderwidth=1,
            borderpad=4
        )
        
        # Update layout for better readability
        fig.update_layout(
            title={
                'text': 'Win Probability vs. Price Curve',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Cost Per Interview (CPI) in $',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': 'Win Probability (%)',
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=40, r=40, t=80, b=60),
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
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(0,0,0,0.2)",
                bordercolor="rgba(255,255,255,0.3)",
                borderwidth=1
            )
        )
        
        # Add explanatory annotation
        fig.add_annotation(
            text="The curve shows how win probability decreases as price increases",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,0.7)")
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating win probability chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_type_distribution_chart(data: pd.DataFrame) -> go.Figure:
    """
    Create a pie chart showing the distribution of won vs lost bids.
    
    Args:
        data (pd.DataFrame): DataFrame with Type column
        
    Returns:
        go.Figure: Plotly figure with type distribution visualization
    """
    try:
        # Count won/lost
        type_counts = data['Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=type_counts['Type'],
            values=type_counts['Count'],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(
                colors=[DARK_THEME_COLORS['won'], DARK_THEME_COLORS['lost']],
                line=dict(color='rgba(255, 255, 255, 0.5)', width=2)
            ),
            textfont=dict(size=14, color='white'),
            hoverinfo='label+value+percent',
            pull=[0.05, 0]
        )])
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Distribution of Won vs Lost Bids',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=14, color='white')
            ),
            annotations=[
                dict(
                    text=f"Total: {type_counts['Count'].sum()} Projects",
                    x=0.5, y=0.5,
                    font=dict(size=16, color='white'),
                    showarrow=False
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating type distribution chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig

def create_cpi_distribution_boxplot(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a boxplot showing the distribution of CPI values for won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with CPI distribution visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add boxplots
        fig.add_trace(go.Box(
            y=won_data['CPI'],
            name='Won Bids',
            boxmean=True,  # Show mean
            marker_color=DARK_THEME_COLORS['won'],
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hoverinfo='y',
            hovertemplate='CPI: $%{y:.2f}<extra>Won Bid</extra>'
        ))
        
        fig.add_trace(go.Box(
            y=lost_data['CPI'],
            name='Lost Bids',
            boxmean=True,  # Show mean
            marker_color=DARK_THEME_COLORS['lost'],
            line=dict(width=2),
            boxpoints='outliers',
            jitter=0.3,
            pointpos=-1.8,
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hoverinfo='y',
            hovertemplate='CPI: $%{y:.2f}<extra>Lost Bid</extra>'
        ))
        
        # Calculate statistics
        won_mean = won_data['CPI'].mean()
        lost_mean = lost_data['CPI'].mean()
        won_median = won_data['CPI'].median()
        lost_median = lost_data['CPI'].median()
        
        # Add annotations for the means
        fig.add_annotation(
            x=0, 
            y=won_mean,
            text=f"Mean: ${won_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
            ax=-40,
            ay=-30,
            bgcolor=DARK_THEME_COLORS['won'],
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            font=dict(color="white", size=12)
        )
        
        fig.add_annotation(
            x=1, 
            y=lost_mean,
            text=f"Mean: ${lost_mean:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="white",
            ax=40,
            ay=-30,
            bgcolor=DARK_THEME_COLORS['lost'],
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            font=dict(color="white", size=12)
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI Distribution: Won vs Lost Bids',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis_title={
                'text': 'Cost Per Interview (CPI) in $',
                'font': {'size': 14, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=40, r=40, t=80, b=40),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI distribution boxplot: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        return fig

def create_cpi_vs_ir_scatter(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of CPI vs IR for won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with scatter plot visualization
    """
    try:
        # Create figure
        fig = go.Figure()
        
        # Add scatter plots for won and lost bids
        fig.add_trace(go.Scatter(
            x=won_data['IR'],
            y=won_data['CPI'],
            mode='markers',
            name='Won Bids',
            marker=dict(
                color=DARK_THEME_COLORS['won'],
                size=10,
                line=dict(width=1, color='white'),
                opacity=0.7
            ),
            hovertemplate='IR: %{x:.1f}%<br>CPI: $%{y:.2f}<extra>Won Bid</extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=lost_data['IR'],
            y=lost_data['CPI'],
            mode='markers',
            name='Lost Bids',
            marker=dict(
                color=DARK_THEME_COLORS['lost'],
                size=10,
                line=dict(width=1, color='white'),
                opacity=0.7
            ),
            hovertemplate='IR: %{x:.1f}%<br>CPI: $%{y:.2f}<extra>Lost Bid</extra>'
        ))
        
        # Add trend lines
        # For won bids
        if len(won_data) > 2:
            try:
                # Simple linear fit
                x = won_data['IR']
                y = won_data['CPI']
                
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Generate x range for line
                x_range = np.linspace(min(x), max(x), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Won Trend',
                    line=dict(color=DARK_THEME_COLORS['won'], width=2, dash='solid'),
                    hoverinfo='skip'
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line for won bids: {e}")
        
        # For lost bids
        if len(lost_data) > 2:
            try:
                # Simple linear fit
                x = lost_data['IR']
                y = lost_data['CPI']
                
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                
                # Generate x range for line
                x_range = np.linspace(min(x), max(x), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Lost Trend',
                    line=dict(color=DARK_THEME_COLORS['lost'], width=2, dash='solid'),
                    hoverinfo='skip'
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line for lost bids: {e}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI vs Incidence Rate (IR)',
                'font': {'size': 20, 'color': 'white'},
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
                'text': 'Cost Per Interview (CPI) in $',
                'font': {'size': 14, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add annotation explaining the relationship
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            text="Lower IR typically leads to higher CPI",
            showarrow=False,
            font=dict(size=14, color="rgba(255,255,255,0.7)"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI vs IR scatter plot: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def create_cpi_efficiency_chart(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing CPI versus a calculated efficiency metric.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        go.Figure: Plotly figure with efficiency visualization
    """
    try:
        # Calculate efficiency metric for won data
        won_data_copy = won_data.copy()
        won_data_copy['Efficiency'] = (
            won_data_copy['IR'] * 
            (1 / np.maximum(1, won_data_copy['LOI'])) * 
            np.log1p(won_data_copy['Completes'])
        )
        
        # Calculate efficiency metric for lost data
        lost_data_copy = lost_data.copy()
        lost_data_copy['Efficiency'] = (
            lost_data_copy['IR'] * 
            (1 / np.maximum(1, lost_data_copy['LOI'])) * 
            np.log1p(lost_data_copy['Completes'])
        )
        
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
                line=dict(width=1, color='white'),
                opacity=0.7
            ),
            hovertemplate='Efficiency: %{x:.2f}<br>CPI: $%{y:.2f}<extra>Won Bid</extra>'
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
                line=dict(width=1, color='white'),
                opacity=0.7
            ),
            hovertemplate='Efficiency: %{x:.2f}<br>CPI: $%{y:.2f}<extra>Lost Bid</extra>'
        ))
        
        # Add trend lines
        # For won bids
        if len(won_data_copy) > 2:
            try:
                # Simple exponential fit
                x = won_data_copy['Efficiency']
                y = won_data_copy['CPI']
                
                # Use log transform for exponential fit
                z = np.polyfit(x, np.log(y + 1), 1)
                # Create function from fit
                p = lambda x_val: np.exp(z[0] * x_val + z[1]) - 1
                
                # Generate x range for line
                x_range = np.linspace(min(x), max(x), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Won Trend',
                    line=dict(color=DARK_THEME_COLORS['won'], width=2, dash='solid'),
                    hoverinfo='skip'
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line for won bids: {e}")
        
        # For lost bids
        if len(lost_data_copy) > 2:
            try:
                # Simple exponential fit
                x = lost_data_copy['Efficiency']
                y = lost_data_copy['CPI']
                
                # Use log transform for exponential fit
                z = np.polyfit(x, np.log(y + 1), 1)
                # Create function from fit
                p = lambda x_val: np.exp(z[0] * x_val + z[1]) - 1
                
                # Generate x range for line
                x_range = np.linspace(min(x), max(x), 100)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name='Lost Trend',
                    line=dict(color=DARK_THEME_COLORS['lost'], width=2, dash='solid'),
                    hoverinfo='skip'
                ))
            except Exception as e:
                logger.warning(f"Could not add trend line for lost bids: {e}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'CPI vs Project Efficiency',
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Efficiency Metric (Higher = More Favorable)',
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': 'Cost Per Interview (CPI) in $',
                'font': {'size': 14, 'color': 'white'}
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            margin=dict(l=40, r=40, t=80, b=40),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)',
                tickprefix='$'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        # Add annotation explaining the metric
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.99, y=0.99,
            xanchor="right", yanchor="top",
            text="Efficiency = IR × (1/LOI) × log(Completes)",
            showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,0.7)"),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
            bordercolor="rgba(255,255,255,0.3)",
            borderwidth=1
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating CPI efficiency chart: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig

def format_chart_for_dark_mode(fig: go.Figure) -> go.Figure:
    """
    Format a Plotly chart for better visibility in dark mode.
    
    Args:
        fig (go.Figure): Input Plotly figure
        
    Returns:
        go.Figure: Formatted figure with dark mode styling
    """
    try:
        # Update the overall layout
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        # Update any additional yaxis if present
        if 'yaxis2' in fig.layout:
            fig.update_layout(
                yaxis2=dict(
                    gridcolor='rgba(255,255,255,0.1)',
                    zerolinecolor='rgba(255,255,255,0.1)'
                )
            )
        
        # Update legend
        if 'legend' in fig.layout:
            fig.update_layout(
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='rgba(0,0,0,0.2)',
                    bordercolor='rgba(255,255,255,0.3)',
                    borderwidth=1
                )
            )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error formatting chart for dark mode: {e}", exc_info=True)
        return fig  # Return original figure if formatting fails

def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                 title: str, categorized: bool = False) -> go.Figure:
    """
    Create an enhanced heatmap visualization optimized for dark mode with improved readability.
    
    Args:
        data (pd.DataFrame): DataFrame with the data
        x_col (str): Column name for x-axis
        y_col (str): Column name for y-axis
        z_col (str): Column name for z values (color)
        title (str): Chart title
        categorized (bool): Whether x and y are category variables
        
    Returns:
        go.Figure: Plotly figure with heatmap visualization
    """
    try:
        if categorized:
            # For categorical variables, pivot the table
            pivot_table = pd.pivot_table(
                data, 
                values=z_col, 
                index=y_col, 
                columns=x_col, 
                aggfunc='mean'
            )
            
            # Create heatmap with Plotly
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdBu_r',  # Red-White-Blue scale reversed
                reversescale=True,
                colorbar=dict(
                    title=dict(text=z_col, font=dict(color='white', size=14)),
                    tickfont=dict(color='white'),
                    outlinecolor='rgba(255,255,255,0.3)',
                    outlinewidth=1
                ),
                hovertemplate=f'{x_col}: %{{x}}<br>{y_col}: %{{y}}<br>{z_col}: %{{z:.2f}}<extra></extra>'
            ))
            
        else:
            # For continuous variables, calculate average z values for bins of x and y
            x_edges = np.linspace(data[x_col].min(), data[x_col].max(), 10)
            y_edges = np.linspace(data[y_col].min(), data[y_col].max(), 10)
            
            # Create empty 2D grid
            grid = np.zeros((len(y_edges)-1, len(x_edges)-1))
            counts = np.zeros((len(y_edges)-1, len(x_edges)-1))
            
            # Fill grid with average z values
            for i, (x_min, x_max) in enumerate(zip(x_edges[:-1], x_edges[1:])):
                for j, (y_min, y_max) in enumerate(zip(y_edges[:-1], y_edges[1:])):
                    mask = (data[x_col] >= x_min) & (data[x_col] < x_max) & \
                           (data[y_col] >= y_min) & (data[y_col] < y_max)
                    
                    if mask.sum() > 0:
                        grid[j, i] = data.loc[mask, z_col].mean()
                        counts[j, i] = mask.sum()
                    else:
                        grid[j, i] = np.nan
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=grid,
                x=[(x_edges[i] + x_edges[i+1])/2 for i in range(len(x_edges)-1)],
                y=[(y_edges[i] + y_edges[i+1])/2 for i in range(len(y_edges)-1)],
                colorscale='RdBu_r',
                reversescale=True,
                colorbar=dict(
                    title=dict(text=z_col, font=dict(color='white', size=14)),
                    tickfont=dict(color='white'),
                    outlinecolor='rgba(255,255,255,0.3)',
                    outlinewidth=1
                ),
                hovertemplate=f'{x_col}: %{{x:.2f}}<br>{y_col}: %{{y:.2f}}<br>{z_col}: %{{z:.2f}}<extra></extra>'
            ))
        
        # Update layout for better readability
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': 'white'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': x_col,
                'font': {'size': 14, 'color': 'white'}
            },
            yaxis_title={
                'text': y_col,
                'font': {'size': 14, 'color': 'white'}
            },
            margin=dict(l=50, r=40, t=80, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                zerolinecolor='rgba(255,255,255,0.1)'
            )
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}", exc_info=True)
        # Return a minimal figure if error occurs
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        return fig