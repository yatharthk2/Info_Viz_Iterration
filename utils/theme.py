"""
Theme utilities for the CPI Analysis & Prediction Dashboard.
Provides custom styling and theming functionality.
"""

import streamlit as st
from typing import Dict, List, Any

def apply_custom_theme():
    """
    Apply custom styling to the dashboard using CSS.
    """
    # Custom CSS for high-contrast dark theme enhancements
    custom_css = """
    <style>
    /* Improve text contrast */
    .stTextInput > label, .stSelectbox > label, .stSlider > label {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Enhance header contrast */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    /* Make metrics more visible */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }
    
    /* Enhance metric delta styling */
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Improve button contrast */
    .stButton button {
        font-weight: 600 !important;
        border-radius: 4px !important;
    }
    
    /* Emphasize expandable sections */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #ffffff !important;
    }
    
    /* Card-like effect for better section delineation */
    .element-container:has(.block-container) {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    /* Enhanced tooltip for better readability */
    .tooltip-content {
        background-color: #222a36 !important;
        border: 1px solid #4e79a7 !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    
    /* Make sure table headers are more visible */
    .dataframe th {
        background-color: #4e79a7 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    /* Table row styling for better readability */
    .dataframe tr:nth-child(even) {
        background-color: #1e2130 !important;
    }
    
    /* Better sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid #2c313d !important;
    }
    
    /* Enhanced sidebar header */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        padding-top: 1rem !important;
        color: #ffffff !important;
        font-weight: 800 !important;
    }
    
    /* Warning/error message styling */
    .stAlert {
        border-radius: 4px !important;
        border-width: 2px !important;
    }
    
    /* Give more emphasis to metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2c313d;
        border-radius: 5px;
        padding: 10px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """
    
    # Apply the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

def highlight_text(text: str, style: str = "info") -> str:
    """
    Create styled highlighted text for better visual cues.
    
    Args:
        text (str): Text to highlight
        style (str): Style to apply (info, success, warning, danger)
        
    Returns:
        str: HTML for styled text
    """
    styles = {
        "info": "background-color: #4e79a7; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-weight: bold;",
        "success": "background-color: #59a14f; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-weight: bold;",
        "warning": "background-color: #edc949; color: black; padding: 0.2rem 0.5rem; border-radius: 3px; font-weight: bold;",
        "danger": "background-color: #e15759; color: white; padding: 0.2rem 0.5rem; border-radius: 3px; font-weight: bold;"
    }
    
    return f'<span style="{styles.get(style, styles["info"])}">{text}</span>'

def format_metric_value(value: float, prefix: str = "", suffix: str = "", decimal_places: int = 2) -> str:
    """
    Format a value for display as a metric.
    
    Args:
        value (float): The value to format
        prefix (str): Prefix such as "$" or "€"
        suffix (str): Suffix such as "%" or " min"
        decimal_places (int): Number of decimal places
        
    Returns:
        str: Formatted metric value
    """
    format_str = f"{prefix}{{:.{decimal_places}f}}{suffix}"
    return format_str.format(value)
