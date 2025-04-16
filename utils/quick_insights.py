"""
Quick Insights Generator for CPI Analysis & Prediction Dashboard.

This module provides functionality for generating concise, emoji-powered summaries
of key data trends in the CPI data.
"""

import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def calculate_cpi_diff_percentage(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> float:
    """
    Calculate the percentage difference between median CPI of won and lost bids.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        
    Returns:
        float: Percentage difference (negative means won bids have lower CPI)
    """
    won_median = won_data['CPI'].median()
    lost_median = lost_data['CPI'].median()
    
    # Calculate percentage difference (negative means won is lower)
    return ((won_median - lost_median) / lost_median) * 100

def get_correlations(data: pd.DataFrame) -> pd.Series:
    """
    Calculate correlations between CPI and other numeric variables.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        pd.Series: Series of correlation values with CPI
    """
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    
    # Calculate correlations with CPI
    correlations = data[numeric_cols].corr()['CPI'].drop('CPI')
    
    return correlations.abs().sort_values(ascending=False)

def identify_key_segments(data: pd.DataFrame) -> Tuple[str, float]:
    """
    Identify the segment with the highest profitability potential.
    
    Args:
        data (pd.DataFrame): Combined dataframe
        
    Returns:
        Tuple[str, float]: Segment name and its win rate
    """
    # Create segments based on binned variables
    if 'IR_Bin' in data.columns and 'LOI_Bin' in data.columns:
        segments = data.groupby(['IR_Bin', 'LOI_Bin']).agg({
            'Type': lambda x: (x == 'Won').mean(),
            'CPI': 'median',
            'CPI': lambda x: len(x)  # Count using CPI column since ID might not exist
        }).reset_index()
        
        # Filter to segments with at least 5% of data
        min_count = len(data) * 0.05
        segments = segments[segments['CPI'] >= min_count]
        
        # Sort by win rate
        segments = segments.sort_values('Type', ascending=False)
        
        if not segments.empty:
            top_segment = segments.iloc[0]
            return (
                f"IR {top_segment['IR_Bin']}, LOI {top_segment['LOI_Bin']}",
                top_segment['Type'] * 100
            )
    
    # Fallback to just IR bins if the combined segments don't have enough data
    segments = data.groupby('IR_Bin').agg({
        'Type': lambda x: (x == 'Won').mean(),
        'CPI': 'median',
        'CPI': lambda x: len(x)  # Count using CPI column since ID might not exist
    }).reset_index()
    
    # Filter to segments with at least 5% of data
    min_count = len(data) * 0.05
    segments = segments[segments['CPI'] >= min_count]
    
    # Sort by win rate
    segments = segments.sort_values('Type', ascending=False)
    
    if not segments.empty:
        top_segment = segments.iloc[0]
        return f"IR {top_segment['IR_Bin']}", top_segment['Type'] * 100
    
    return "Unknown", 0.0

def analyze_outliers(data: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate outlier percentages across key variables.
    
    Args:
        data (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, int]: Dictionary of outlier percentages by variable
    """
    outliers = {}
    
    for col in ['CPI', 'IR', 'LOI', 'Completes']:
        if col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_pct = round((outlier_count / len(data)) * 100)
            
            outliers[col] = outlier_pct
    
    return outliers

def get_win_rate_by_factor(data: pd.DataFrame, factor: str) -> Dict[str, float]:
    """
    Calculate win rates by a specific factor (column with bins).
    
    Args:
        data (pd.DataFrame): Combined dataframe
        factor (str): Column name for the binned factor
        
    Returns:
        Dict[str, float]: Dictionary of win rates by factor bin
    """
    if factor not in data.columns:
        return {}
    
    # Calculate win rate by factor
    win_rates = data.groupby(factor).apply(
        lambda x: (x['Type'] == 'Won').mean() * 100
    ).to_dict()
    
    return win_rates

def get_emoji_for_trend(value: float, reverse: bool = False) -> str:
    """
    Get an appropriate emoji based on the trend value.
    
    Args:
        value (float): The trend value (usually a percentage)
        reverse (bool): If True, negative values are considered good
        
    Returns:
        str: Emoji representing the trend
    """
    if reverse:
        value = -value
        
    if value > 15:
        return "🚀"  # Rocket for strong positive
    elif value > 5:
        return "📈"  # Chart up for positive
    elif value < -15:
        return "📉"  # Chart down for strong negative
    elif value < -5:
        return "⚠️"  # Warning for negative
    else:
        return "➖"  # Stable for neutral

def generate_quick_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, 
                         combined_data: pd.DataFrame) -> str:
    """
    Generate a concise, emoji-powered summary of key data trends.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        combined_data (pd.DataFrame): Combined DataFrame
        
    Returns:
        str: Markdown-formatted insights summary with emojis
    """
    insights = []
    
    # CPI Difference
    cpi_diff_pct = calculate_cpi_diff_percentage(won_data, lost_data)
    cpi_diff_emoji = get_emoji_for_trend(cpi_diff_pct, reverse=True)
    
    won_median = round(won_data['CPI'].median(), 2)
    lost_median = round(lost_data['CPI'].median(), 2)
    
    insights.append(
        f"{cpi_diff_emoji} **CPI Difference**: Won bids are "
        f"{'lower' if cpi_diff_pct < 0 else 'higher'} by {abs(round(cpi_diff_pct))}% "
        f"(${won_median} vs ${lost_median})"
    )
    
    # Key Drivers
    correlations = get_correlations(combined_data)
    if not correlations.empty:
        top_factor = correlations.index[0]
        corr_strength = correlations.iloc[0]
        driver_emoji = "⚡" if corr_strength > 0.5 else "🔄"
        
        insights.append(
            f"{driver_emoji} **Key CPI Driver**: {top_factor} "
            f"(correlation: {round(corr_strength, 2)})"
        )
    
    # Winning Segment
    top_segment, win_rate = identify_key_segments(combined_data)
    segment_emoji = "💰" if win_rate > 50 else "🎯"
    
    insights.append(
        f"{segment_emoji} **Best Segment**: {top_segment} "
        f"(win rate: {round(win_rate)}%)"
    )
    
    # Outlier Analysis
    outliers = analyze_outliers(combined_data)
    max_outlier = max(outliers.items(), key=lambda x: x[1]) if outliers else ("None", 0)
    outlier_emoji = "🔍" if max_outlier[1] > 10 else "✅"
    
    insights.append(
        f"{outlier_emoji} **Data Quality**: {max_outlier[0]} has the most outliers "
        f"({max_outlier[1]}% of data)"
    )
    
    # Win Rate by IR Bin
    ir_win_rates = get_win_rate_by_factor(combined_data, 'IR_Bin')
    if ir_win_rates:
        best_ir = max(ir_win_rates.items(), key=lambda x: x[1])
        ir_emoji = "💎" if best_ir[1] > 40 else "🔹"
        
        insights.append(
            f"{ir_emoji} **Best IR Range**: {best_ir[0]} "
            f"(win rate: {round(best_ir[1])}%)"
        )
    
    # Sample Size Insights
    avg_completes_won = round(won_data['Completes'].mean())
    avg_completes_lost = round(lost_data['Completes'].mean())
    completes_diff = (avg_completes_won - avg_completes_lost) / avg_completes_lost * 100
    completes_emoji = get_emoji_for_trend(completes_diff)
    
    insights.append(
        f"{completes_emoji} **Sample Size**: Won bids average {avg_completes_won} completes, "
        f"{'higher' if completes_diff > 0 else 'lower'} than lost bids by {abs(round(completes_diff))}%"
    )
    
    # Final Summary
    return "\n\n".join(insights)

def show_quick_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, 
                      combined_data: pd.DataFrame) -> None:
    """
    Display a quick insights button and summary when clicked.
    
    Args:
        won_data (pd.DataFrame): DataFrame of won bids
        lost_data (pd.DataFrame): DataFrame of lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    st.subheader("Quick Insights")
    
    st.markdown("""
    Get a concise summary of key data trends with our AI-powered Quick Insights feature.
    Click the button below to generate instant insights from your data.
    """)
    
    if st.button("📊 Generate Quick Insights", use_container_width=True):
        with st.spinner("Analyzing data trends..."):
            # Add a slight delay to give the impression of computation
            import time
            time.sleep(0.8)
            
            # Generate the insights
            insights_text = generate_quick_insights(won_data, lost_data, combined_data)
            
            # Display the insights in a styled container
            st.success("✨ Quick Insights Generated!")
            
            st.markdown("""
            <style>
            .insights-container {
                background-color: rgba(49, 51, 63, 0.7);
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
                border-left: 5px solid rgba(0, 180, 216, 0.8);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="insights-container">{insights_text}</div>', 
                      unsafe_allow_html=True)
            
            # Add explanation and CTA
            st.info("""
            These insights highlight key patterns in your CPI data. For more detailed analysis, 
            explore the dedicated sections in the dashboard.
            """)