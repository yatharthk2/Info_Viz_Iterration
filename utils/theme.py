"""
Theme utilities for the CPI Analysis & Prediction Dashboard.
Provides functions for applying custom themes and styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dark theme colors
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
    "borders": "#333333",
}

def apply_custom_theme() -> None:
    """
    Apply a custom theme to the Streamlit application with high-contrast colors for better readability.
    The theme is optimized for data visualization with dark backgrounds.
    """
    try:
        # Add custom CSS
        st.markdown("""
        <style>
        /* Base text styling */
        html, body, [class*="css"] {
            font-family: 'Source Sans Pro', sans-serif;
            color: #FAFAFA;
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-12oz5g7 {
            background-color: #1E2130;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5 {
            color: #FFFFFF !important;
            font-weight: 600 !important;
        }
        h1 {
            font-size: 2.2em !important;
            margin-bottom: 0.5em !important;
            border-bottom: 2px solid #4e79a7;
            padding-bottom: 0.3em;
        }
        h2 {
            font-size: 1.8em !important;
            margin-top: 1em !important;
        }
        h3 {
            font-size: 1.4em !important;
            margin-top: 1em !important;
            color: #7EB3FF !important;
        }
        
        /* Statistics */
        .metric-value {
            font-size: 2.5em !important;
            font-weight: bold !important;
            color: #FFFFFF !important;
        }
        .metric-label {
            font-size: 1em !important;
            color: #BBBBBB !important;
        }
        
        /* Enhance container backgrounds */
        div.stBlock {
            background-color: #1E2130;
            padding: 1em;
            border-radius: 5px;
        }
        
        /* Better button styling */
        .stButton>button {
            background-color: #4e79a7 !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.5em 1em !important;
            border-radius: 4px !important;
            transition: all 0.3s ease !important;
        }
        .stButton>button:hover {
            background-color: #3A5980 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        }
        .stButton>button:active {
            transform: translateY(1px) !important;
        }
        
        /* Slider styling for better visibility */
        .stSlider {
            padding-top: 0.5em;
            padding-bottom: 1em;
        }
        .stSlider > div > div {
            background-color: rgba(78, 121, 167, 0.3) !important;
        }
        .stSlider > div > div > div > div {
            background-color: #4e79a7 !important;
        }
        
        /* Table styling */
        .dataframe {
            border: 1px solid #333333 !important;
        }
        .dataframe th {
            background-color: #1E2130 !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            border: 1px solid #333333 !important;
        }
        .dataframe td {
            background-color: #111111 !important;
            color: #DDDDDD !important;
            border: 1px solid #333333 !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(78, 121, 167, 0.1) !important;
            border-radius: 4px !important;
        }
        .streamlit-expanderHeader:hover {
            background-color: rgba(78, 121, 167, 0.2) !important;
        }
        
        /* Radio button styling */
        .stRadio > div {
            background-color: #1E2130 !important;
            border-radius: 4px !important;
            padding: 0.5em !important;
        }
        
        /* Custom class for dark cards */
        .dark-card {
            background-color: #1E2130;
            border-radius: 5px;
            padding: 1.5em;
            margin-bottom: 1em;
            border-left: 4px solid #4e79a7;
        }
        
        /* Custom class for metric containers */
        .metric-container {
            background-color: rgba(30, 33, 48, 0.7);
            border-radius: 5px;
            padding: 1em;
            text-align: center;
            border-bottom: 3px solid #4e79a7;
        }
        
        /* Better select box styling */
        .stSelectbox > div > div {
            background-color: #1E2130 !important;
            color: #FFFFFF !important;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1E2130 !important;
            color: #BBBBBB !important;
            padding: 10px 20px;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4e79a7 !important;
            color: #FFFFFF !important;
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #BBBBBB;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #1E2130;
            color: #FFFFFF;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Chart area modifications */
        .js-plotly-plot {
            background-color: transparent !important;
        }
        
        /* Focus indicators for accessibility */
        a:focus, button:focus, input:focus, select:focus, textarea:focus {
            outline: 2px solid #7EB3FF !important;
            outline-offset: 2px !important;
        }
        
        /* Better horizontal rule styling */
        hr {
            border-top: 1px solid #333333 !important;
            margin: 1.5em 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        logger.info("Applied custom dark high-contrast theme")
    
    except Exception as e:
        logger.error(f"Error applying custom theme: {e}", exc_info=True)

def create_metric_card(title: str, value: Any, unit: str = "", delta: float = None, 
                     interpret: str = None, help_text: str = None) -> None:
    """
    Create a styled metric card for displaying important KPIs.
    
    Args:
        title (str): Metric title
        value (Any): Metric value
        unit (str, optional): Unit of measure (e.g., "$", "%"). Defaults to "".
        delta (float, optional): Change value for delta indicator. Defaults to None.
        interpret (str, optional): Interpretation text. Defaults to None.
        help_text (str, optional): Help text for tooltip. Defaults to None.
    """
    st.markdown(f"""
    <div class="metric-container">
        <div style="font-size:0.9em; color:#BBBBBB; margin-bottom:0.3em;">{title}</div>
        <div style="font-size:2em; font-weight:bold; color:#FFFFFF;">{value}{unit}</div>
        
        {f'<div style="margin-top:0.3em; color:{"#52BC9F" if delta > 0 else "#E15759"};"><span>{"▲" if delta > 0 else "▼"}</span> {abs(delta)}{unit}</div>' if delta is not None else ''}
        
        {f'<div style="font-size:0.8em; color:#BBBBBB; margin-top:0.5em;">{interpret}</div>' if interpret else ''}
    </div>
    """, unsafe_allow_html=True)
    
    if help_text:
        st.caption(help_text)

def create_section_header(title: str, description: str = None, icon: str = None) -> None:
    """
    Create a styled section header with optional description and icon.
    
    Args:
        title (str): Section title
        description (str, optional): Section description. Defaults to None.
        icon (str, optional): Icon character (emoji). Defaults to None.
    """
    if icon:
        title = f"{icon} {title}"
    
    st.markdown(f"""
    <div style="border-bottom: 2px solid #4e79a7; margin-bottom: 1em; padding-bottom: 0.5em;">
        <h2 style="margin-bottom: 0.2em;">{title}</h2>
        {f'<p style="color: #BBBBBB;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def format_value(value: float, format_type: str = 'currency', precision: int = 2) -> str:
    """
    Format a numeric value with the appropriate formatting based on type.
    
    Args:
        value (float): Value to format
        format_type (str, optional): Format type ('currency', 'percent', 'number'). Defaults to 'currency'.
        precision (int, optional): Decimal precision. Defaults to 2.
    
    Returns:
        str: Formatted value as a string
    """
    try:
        if format_type == 'currency':
            return f"${value:,.{precision}f}"
        elif format_type == 'percent':
            return f"{value:.{precision}f}%"
        else:  # 'number'
            return f"{value:,.{precision}f}"
    except Exception as e:
        logger.error(f"Error formatting value: {e}")
        return str(value)  # Fallback to string conversion