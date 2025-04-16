"""
Theme utilities for the CPI Analysis & Prediction Dashboard.
Provides functions for applying custom themes and styling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced color system
COLOR_SYSTEM = {
    # Primary palette
    'PRIMARY': {
        'MAIN': '#4e79a7',       # Primary blue - headers, primary elements
        'LIGHT': '#7EB3FF',      # Lighter blue - highlights
        'DARK': '#3A5980',       # Darker blue - emphasis
        'CONTRAST': '#FFFFFF'    # White - text on dark backgrounds
    },
    
    # Accent colors
    'ACCENT': {
        'BLUE': '#4e79a7',       # Primary blue - won bids
        'ORANGE': '#f28e2b',     # Orange - lost bids
        'GREEN': '#52BC9F',      # Green - positive indicators
        'RED': '#E15759',        # Red - negative indicators
        'PURPLE': '#b07aa1',     # Purple - predictions
        'YELLOW': '#F6C85F'      # Yellow - warnings, highlights
    },
    
    # Neutral tones
    'NEUTRAL': {
        'WHITE': '#FFFFFF',
        'LIGHTEST': '#F8F9FA',
        'LIGHTER': '#E9ECEF',
        'LIGHT': '#BBBBBB',
        'MEDIUM': '#888888',
        'DARK': '#555555',
        'DARKER': '#333333',
        'DARKEST': '#1E2130',
        'BLACK': '#0E1117'
    },
    
    # Semantic colors (for specific meanings)
    'SEMANTIC': {
        'SUCCESS': '#52BC9F',
        'WARNING': '#F6C85F',
        'ERROR': '#E15759',
        'INFO': '#4e79a7'
    },
    
    # Chart-specific colors 
    'CHARTS': {
        'WON': '#4e79a7',        # Blue
        'LOST': '#f28e2b',       # Orange
        'WON_TRANS': 'rgba(78, 121, 167, 0.7)',
        'LOST_TRANS': 'rgba(242, 142, 43, 0.7)',
        'GRADIENT_1': '#3A5980',
        'GRADIENT_2': '#7EB3FF',
        'GRADIENT_3': '#FFFFFF',
        # Series colors for charts (consistent color scheme)
        'SERIES1': '#4E79A7',    # Blue
        'SERIES2': '#F28E2B',    # Orange
        'SERIES3': '#E15759',    # Red
        'SERIES4': '#76B7B2',    # Teal
        'SERIES5': '#59A14F',    # Green
        'SERIES6': '#EDC948',    # Yellow
        'SERIES7': '#B07AA1'     # Purple
    }
}

# Backward compatibility with existing code
DARK_THEME_COLORS = {
    "background": "#0E1117",
    "secondary_background": "#1E2130",
    "primary": COLOR_SYSTEM['PRIMARY']['MAIN'],
    "success": COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
    "warning": COLOR_SYSTEM['SEMANTIC']['WARNING'],
    "error": COLOR_SYSTEM['SEMANTIC']['ERROR'],
    "text": COLOR_SYSTEM['PRIMARY']['CONTRAST'],
    "secondary_text": COLOR_SYSTEM['NEUTRAL']['LIGHT'],
    "highlight": COLOR_SYSTEM['PRIMARY']['LIGHT'],
    "muted": COLOR_SYSTEM['NEUTRAL']['DARK'],
    "borders": COLOR_SYSTEM['NEUTRAL']['DARKER'],
    "won": COLOR_SYSTEM['CHARTS']['WON'],
    "lost": COLOR_SYSTEM['CHARTS']['LOST'],
}

# Typography system
TYPOGRAPHY = {
    'FONT_FAMILY': '"Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'HEADING': {
        'H1': {'size': '2.2rem', 'weight': '700', 'height': '1.2'},
        'H2': {'size': '1.8rem', 'weight': '600', 'height': '1.3'},
        'H3': {'size': '1.4rem', 'weight': '600', 'height': '1.4'},
        'H4': {'size': '1.1rem', 'weight': '600', 'height': '1.5'}
    },
    'BODY': {
        'LARGE': {'size': '1.1rem', 'weight': '400', 'height': '1.5'},
        'NORMAL': {'size': '1rem', 'weight': '400', 'height': '1.5'},
        'SMALL': {'size': '0.875rem', 'weight': '400', 'height': '1.4'}
    }
}

def apply_custom_theme() -> None:
    """
    Apply a custom theme to the Streamlit application with high-contrast colors for better readability.
    The theme is optimized for data visualization with dark backgrounds.
    """
    try:
        # Add custom CSS with enhanced styling using the color and typography systems
        st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
        
        /* Base text styling */
        html, body, [class*="css"] {{
            font-family: {TYPOGRAPHY['FONT_FAMILY']};
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']};
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-12oz5g7 {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5 {{
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            font-weight: 600 !important;
            line-height: 1.2 !important;
        }}
        h1 {{
            font-size: {TYPOGRAPHY['HEADING']['H1']['size']} !important;
            font-weight: {TYPOGRAPHY['HEADING']['H1']['weight']} !important;
            margin-bottom: 0.5em !important;
            border-bottom: 2px solid {COLOR_SYSTEM['PRIMARY']['MAIN']};
            padding-bottom: 0.3em;
        }}
        h2 {{
            font-size: {TYPOGRAPHY['HEADING']['H2']['size']} !important;
            font-weight: {TYPOGRAPHY['HEADING']['H2']['weight']} !important;
            margin-top: 1em !important;
        }}
        h3 {{
            font-size: {TYPOGRAPHY['HEADING']['H3']['size']} !important;
            font-weight: {TYPOGRAPHY['HEADING']['H3']['weight']} !important;
            margin-top: 1em !important;
            color: {COLOR_SYSTEM['PRIMARY']['LIGHT']} !important;
        }}
        
        /* Statistics */
        .metric-value {{
            font-size: 2.5em !important;
            font-weight: bold !important;
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
        }}
        .metric-label {{
            font-size: 1em !important;
            color: {COLOR_SYSTEM['NEUTRAL']['LIGHT']} !important;
        }}
        
        /* Enhance container backgrounds */
        div.stBlock {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']};
            padding: 1em;
            border-radius: 5px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}
        
        /* Better button styling */
        .stButton>button {{
            background-color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.5em 1em !important;
            border-radius: 4px !important;
            transition: all 0.3s ease !important;
        }}
        .stButton>button:hover {{
            background-color: {COLOR_SYSTEM['PRIMARY']['DARK']} !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
        }}
        .stButton>button:active {{
            transform: translateY(1px) !important;
        }}
        
        /* Slider styling for better visibility */
        .stSlider {{
            padding-top: 0.5em;
            padding-bottom: 1em;
        }}
        .stSlider > div > div {{
            background-color: rgba(78, 121, 167, 0.3) !important;
        }}
        .stSlider > div > div > div > div {{
            background-color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
        }}
        
        /* Table styling */
        .dataframe {{
            border: 1px solid {COLOR_SYSTEM['NEUTRAL']['DARKER']} !important;
            border-collapse: separate !important;
            border-spacing: 0 !important;
            border-radius: 4px !important;
            overflow: hidden !important;
        }}
        .dataframe th {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']} !important;
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            font-weight: 600 !important;
            border: 1px solid {COLOR_SYSTEM['NEUTRAL']['DARKER']} !important;
            padding: 0.75rem !important;
            text-align: left !important;
        }}
        .dataframe td {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['BLACK']} !important;
            color: {COLOR_SYSTEM['NEUTRAL']['LIGHTER']} !important;
            border: 1px solid {COLOR_SYSTEM['NEUTRAL']['DARKER']} !important;
            padding: 0.75rem !important;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: rgba(78, 121, 167, 0.1) !important;
            border-radius: 4px !important;
            font-weight: 600 !important;
            color: {COLOR_SYSTEM['PRIMARY']['LIGHT']} !important;
        }}
        .streamlit-expanderHeader:hover {{
            background-color: rgba(78, 121, 167, 0.2) !important;
        }}
        
        /* Radio button styling */
        .stRadio > div {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']} !important;
            border-radius: 4px !important;
            padding: 0.5em !important;
        }}
        
        /* Custom class for dark cards */
        .dark-card {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']};
            border-radius: 5px;
            padding: 1.5em;
            margin-bottom: 1em;
            border-left: 4px solid {COLOR_SYSTEM['PRIMARY']['MAIN']};
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }}
        
        /* Custom class for metric containers */
        .metric-container {{
            background-color: rgba(30, 33, 48, 0.7);
            border-radius: 5px;
            padding: 1.2em;
            text-align: center;
            border-bottom: 3px solid {COLOR_SYSTEM['PRIMARY']['MAIN']};
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-container:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
        }}
        
        /* Better select box styling */
        .stSelectbox > div > div {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']} !important;
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            border-radius: 4px !important;
        }}
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']} !important;
            color: {COLOR_SYSTEM['NEUTRAL']['LIGHT']} !important;
            padding: 10px 20px;
            border-radius: 4px 4px 0 0;
            font-weight: 500 !important;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {COLOR_SYSTEM['PRIMARY']['MAIN']} !important;
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            font-weight: 600 !important;
        }}
        
        /* Tooltip styling */
        .tooltip {{
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted {COLOR_SYSTEM['NEUTRAL']['LIGHT']};
        }}
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 120px;
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']};
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']};
            text-align: center;
            border-radius: 5px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
            font-size: 0.9em;
        }}
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* Chart area modifications */
        .js-plotly-plot {{
            background-color: transparent !important;
            border-radius: 5px !important;
            overflow: hidden !important;
        }}
        
        /* Focus indicators for accessibility */
        a:focus, button:focus, input:focus, select:focus, textarea:focus {{
            outline: 2px solid {COLOR_SYSTEM['PRIMARY']['LIGHT']} !important;
            outline-offset: 2px !important;
        }}
        
        /* Better horizontal rule styling */
        hr {{
            border-top: 1px solid {COLOR_SYSTEM['NEUTRAL']['DARKER']} !important;
            margin: 1.5em 0 !important;
        }}
        
        /* Card-like containers */
        .css-card {{
            background-color: {COLOR_SYSTEM['NEUTRAL']['DARKEST']};
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-top: 3px solid {COLOR_SYSTEM['PRIMARY']['MAIN']};
            transition: transform 0.2s ease;
        }}
        
        .css-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }}
        
        /* Make links more visible */
        a {{
            color: {COLOR_SYSTEM['PRIMARY']['LIGHT']} !important;
            text-decoration: none !important;
            transition: color 0.2s !important;
        }}
        
        a:hover {{
            color: {COLOR_SYSTEM['PRIMARY']['CONTRAST']} !important;
            text-decoration: underline !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        logger.info("Applied custom dark high-contrast theme")
    
    except Exception as e:
        logger.error(f"Error applying custom theme: {e}", exc_info=True)

def create_metric_card(title: str, value: Any, unit: str = "", delta: Optional[float] = None, 
                     interpret: Optional[str] = None, help_text: Optional[str] = None) -> None:
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

def create_section_header(title: str, description: Optional[str] = None, icon: Optional[str] = None) -> None:
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

def format_chart_for_dark_mode(fig, title: Optional[str] = None, height: int = 500) -> Any:
    """
    Format a Plotly chart for dark mode with consistent styling.
    
    Args:
        fig: Plotly figure object
        title (str, optional): Chart title. Defaults to None.
        height (int, optional): Chart height in pixels. Defaults to 500.
        
    Returns:
        Plotly figure with dark mode styling applied
    """
    fig.update_layout(
        # Basic layout
        template="plotly_dark",
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        
        # Background colors
        paper_bgcolor=COLOR_SYSTEM['NEUTRAL']['BLACK'],
        plot_bgcolor='rgba(0,0,0,0)',
        
        # Title configuration
        title=dict(
            text=title if title else "",
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=20,
                color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
            ),
            x=0.01,
            xanchor='left',
            y=0.98,
            yanchor='top'
        ),
        
        # Font configuration for all text elements
        font=dict(
            family=TYPOGRAPHY['FONT_FAMILY'],
            size=14,
            color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
        ),
        
        # Legend configuration
        legend=dict(
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
            ),
            bgcolor='rgba(0,0,0,0.1)',
            bordercolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            borderwidth=1
        ),
        
        # Axes configuration
        xaxis=dict(
            showgrid=True,
            gridcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            gridwidth=1,
            showline=True,
            linecolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['NEUTRAL']['LIGHT']
            ),
            title=dict(
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=13,
                    color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
                )
            ),
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            gridwidth=1,
            showline=True,
            linecolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            tickfont=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['NEUTRAL']['LIGHT']
            ),
            title=dict(
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=13,
                    color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
                )
            ),
            zeroline=False
        ),
        
        # Hoverlabel configuration
        hoverlabel=dict(
            bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
            bordercolor=COLOR_SYSTEM['PRIMARY']['MAIN'],
            font=dict(
                family=TYPOGRAPHY['FONT_FAMILY'],
                size=12,
                color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
            )
        )
    )
    
    return fig