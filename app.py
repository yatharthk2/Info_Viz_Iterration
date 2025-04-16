"""
CPI Analysis & Prediction Dashboard

A Streamlit dashboard for analyzing the Cost Per Interview (CPI) between won and lost bids
and predicting optimal pricing for new bids.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Any, Optional

# Import components
from components.overview import show_overview
from components.prediction import show_prediction
from components.insights import show_insights

# Import data processing utilities
from data_processor import load_data, clean_data, engineer_features

# Import theme utilities
from utils.theme import apply_custom_theme

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config for the app
st.set_page_config(
    page_title="CPI Analysis & Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    """Main function to run the Streamlit app."""
    try:
        # Apply custom theming
        apply_custom_theme()
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        
        # Add app info and selection
        page = st.sidebar.radio(
            "Select Section:",
            ["Overview", "CPI Prediction", "Insights & Recommendations"]
        )
        
        # Sidebar filters
        st.sidebar.title("Data Filters")
        
        # Toggle for extreme value filtering
        filter_extremes = st.sidebar.checkbox("Filter out extreme values", value=True)
        
        # Add sidebar info
        with st.sidebar.expander("About this Dashboard"):
            st.markdown("""
            This dashboard analyzes Cost Per Interview (CPI) data to help optimize pricing
            strategy for market research bids. Key features:
            
            - **Overview**: High-level analysis of CPI patterns
            - **CPI Prediction**: ML-based price prediction tool
            - **Insights**: Strategic recommendations
            
            Data is filtered by default to remove extreme outliers.
            """)
        
        # Load and process data
        try:
            # In a real application, this would load from a database or uploaded file
            # For this example, we'll mock a simple loading function
            raw_data = load_data()
            
            if raw_data is None or raw_data.empty:
                st.error("No data available. Please check your data source or upload a file.")
                return
                
            # Clean data
            cleaned_data = clean_data(raw_data, filter_extremes)
            
            # Split into won/lost categories
            won_data = cleaned_data[cleaned_data['Type'] == 'Won'].copy()
            lost_data = cleaned_data[cleaned_data['Type'] == 'Lost'].copy()
            
            # Basic checks
            if len(won_data) == 0 or len(lost_data) == 0:
                st.warning("One or more bid categories has no data. Analysis may be limited.")
            
            # Engineer features for modeling
            combined_data_engineered = engineer_features(cleaned_data)
        
        except Exception as e:
            logger.error(f"Error loading or processing data: {e}", exc_info=True)
            st.error(f"Error loading or processing data: {str(e)}")
            st.stop()
            
        # Display the selected page
        if page == "Overview":
            show_overview(won_data, lost_data, cleaned_data)
        elif page == "CPI Prediction":
            show_prediction(combined_data_engineered, won_data, lost_data)
        elif page == "Insights & Recommendations":
            show_insights(won_data, lost_data, cleaned_data)
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
