"""
Overview component for the CPI Analysis & Prediction Dashboard.
Displays a high-level summary of the data and key metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# Import visualization utilities
from utils.visualization import (
    create_cpi_comparison_chart,
    create_feature_importance_chart,
    create_prediction_comparison_chart,
    create_heatmap
)

# Import theme utilities
from utils.theme import create_section_header, format_value

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview dashboard showing key metrics and charts.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("CPI Analysis Dashboard: Overview")
        
        # Introduction with enhanced styling and help tooltips
        st.markdown("""
        This dashboard analyzes the Cost Per Interview (CPI) between won and lost bids 
        to identify meaningful differences. The three main factors that influence CPI are:
        """)
        
        # Add highlighted key terms with help tooltips
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; text-align:center;">
                <h3 style="color:#4e79a7;">IR</h3>
                <p>Incidence Rate</p>
                <p style="font-size:0.8em;">The percentage of people who qualify for a survey</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; text-align:center;">
                <h3 style="color:#4e79a7;">LOI</h3>
                <p>Length of Interview</p>
                <p style="font-size:0.8em;">How long the survey takes to complete in minutes</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; text-align:center;">
                <h3 style="color:#4e79a7;">Completes</h3>
                <p>Sample Size</p>
                <p style="font-size:0.8em;">The number of completed surveys required</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add navigation instructions
        st.info("Use the navigation menu on the left to explore different analyses and tools.")
        
        # Help expander with more details
        with st.expander("📖 How to use this dashboard"):
            st.markdown("""
            ### Dashboard Guide
            
            This overview page provides a high-level summary of your CPI data, showing the key differences
            between won and lost bids. Here's how to make the most of it:
            
            1. **Key Metrics Section**: Shows the average values for CPI, IR, and LOI, helping you quickly understand the differences between won and lost bids.
            
            2. **Data Distribution Charts**: Visualizes how won and lost bids are distributed, providing context for your analysis.
            
            3. **CPI vs. Factors Charts**: Shows how CPI relates to other factors, helping identify patterns and relationships.
            
            4. **Navigation**: Use the sidebar to switch between different dashboard sections for deeper analysis.
            
            5. **Filtering**: Toggle the "Filter out extreme values" option in the sidebar to include or exclude outliers.
            
            For more detailed analysis, use the CPI Analysis, CPI Prediction, and Insights sections accessible from the sidebar.
            """)
        
        # Key metrics section with enhanced styling
        st.header("Key Metrics")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 3px'>", unsafe_allow_html=True)
        
        # Calculate metrics with error handling
        won_mean_cpi = won_data['CPI'].mean() if not won_data.empty else 0
        lost_mean_cpi = lost_data['CPI'].mean() if not lost_data.empty else 0
        cpi_diff = lost_mean_cpi - won_mean_cpi
        cpi_diff_pct = (cpi_diff / won_mean_cpi * 100) if won_mean_cpi > 0 else 0
        
        won_median_cpi = won_data['CPI'].median() if not won_data.empty else 0
        lost_median_cpi = lost_data['CPI'].median() if not lost_data.empty else 0
        
        won_mean_ir = won_data['IR'].mean() if not won_data.empty else 0
        lost_mean_ir = lost_data['IR'].mean() if not lost_data.empty else 0
        ir_diff = lost_mean_ir - won_mean_ir
        
        won_mean_loi = won_data['LOI'].mean() if not won_data.empty else 0
        lost_mean_loi = lost_data['LOI'].mean() if not lost_data.empty else 0
        loi_diff = lost_mean_loi - won_mean_loi
        
        # Enhanced metric display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("CPI Metrics")
            
            # Enhanced CPI metrics with contextual colors
            metric_color = "#52bca3" if cpi_diff > 0 else "#e15759"
            
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average CPI - Won</p>
                <p style="font-size:1.8em; font-weight:bold; color:#52bca3; margin:0;">${won_mean_cpi:.2f}</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average CPI - Lost</p>
                <p style="font-size:1.8em; font-weight:bold; color:#e15759; margin:0;">${lost_mean_cpi:.2f}</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; background-color:rgba(78, 121, 167, 0.1);">
                <p style="font-size:1.1em; margin-bottom:5px;">CPI Difference</p>
                <p style="font-size:1.8em; font-weight:bold; color:{metric_color}; margin:0;">${cpi_diff:.2f}</p>
                <p style="font-size:1.1em; color:{metric_color};">{cpi_diff_pct:+.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("IR Metrics")
            
            # Enhanced IR metrics with contextual colors
            metric_color = "#52bca3" if ir_diff < 0 else "#e15759"
            
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average IR - Won</p>
                <p style="font-size:1.8em; font-weight:bold; color:#52bca3; margin:0;">{won_mean_ir:.2f}%</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average IR - Lost</p>
                <p style="font-size:1.8em; font-weight:bold; color:#e15759; margin:0;">{lost_mean_ir:.2f}%</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; background-color:rgba(78, 121, 167, 0.1);">
                <p style="font-size:1.1em; margin-bottom:5px;">IR Difference</p>
                <p style="font-size:1.8em; font-weight:bold; color:{metric_color}; margin:0;">{ir_diff:.2f}%</p>
                <p style="font-size:1.1em; color:{metric_color};">{ir_diff:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.subheader("LOI Metrics")
            
            # Enhanced LOI metrics with contextual colors
            metric_color = "#52bca3" if loi_diff < 0 else "#e15759"
            
            st.markdown(f"""
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average LOI - Won</p>
                <p style="font-size:1.8em; font-weight:bold; color:#52bca3; margin:0;">{won_mean_loi:.2f} min</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; margin-bottom:10px;">
                <p style="font-size:1.1em; margin-bottom:5px;">Average LOI - Lost</p>
                <p style="font-size:1.8em; font-weight:bold; color:#e15759; margin:0;">{lost_mean_loi:.2f} min</p>
            </div>
            
            <div style="border:1px solid #4e79a7; border-radius:5px; padding:10px; background-color:rgba(78, 121, 167, 0.1);">
                <p style="font-size:1.1em; margin-bottom:5px;">LOI Difference</p>
                <p style="font-size:1.8em; font-weight:bold; color:{metric_color}; margin:0;">{loi_diff:.2f} min</p>
                <p style="font-size:1.1em; color:{metric_color};">{loi_diff:+.2f} min</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Overview charts with enhanced styling
        st.header("Data Distribution")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create and display pie chart
            fig = create_type_distribution_chart(combined_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation tooltip
            with st.expander("📊 Understanding this chart"):
                st.markdown("""
                This pie chart shows the proportion of won vs. lost bids in your dataset.
                
                **Why it matters**: The distribution helps you understand your historical win rate
                and provides context for the analysis. A balanced dataset with sufficient data in
                both categories provides more reliable insights.
                """)
        
        with col2:
            # Create and display CPI boxplot
            fig = create_cpi_distribution_boxplot(won_data, lost_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation tooltip
            with st.expander("📊 Understanding this chart"):
                st.markdown("""
                This boxplot shows the distribution of CPI values for both won and lost bids.
                
                **Key elements**:
                - The box represents the middle 50% of values (interquartile range)
                - The line inside the box shows the median
                - The diamond marker shows the mean
                - Points outside the whiskers are potential outliers
                
                **Why it matters**: This visualization helps you understand the typical range of CPI
                values for won vs. lost bids and identify any overlap or clear separation between them.
                """)
        
        # CPI vs IR scatter plot with enhanced styling
        st.header("CPI vs Key Factors")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # IR vs CPI relationship
        st.subheader("CPI vs Incidence Rate (IR)")
        fig = create_cpi_vs_ir_scatter(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### What This Chart Shows
            
            The scatter plot visualizes how CPI (Cost Per Interview) relates to Incidence Rate (IR). 
            Each dot represents a project, with blue dots showing won bids and orange dots showing lost bids.
            
            ### Key Insights
            
            1. **Inverse Relationship**: Generally, as IR increases, CPI decreases. This makes sense because higher
               incidence rates mean it's easier to find qualified respondents.
            
            2. **Won vs Lost Gap**: Notice that lost bids (orange) tend to have higher CPIs than won bids (blue)
               at similar IR levels. This suggests that pricing competitiveness is a key factor in winning bids.
            
            3. **Trend Lines**: The lines show the general trend for each bid type. The gap between these lines
               represents the typical pricing differential between won and lost bids.
            """)
        
        # Add CPI efficiency chart (new visualization)
        st.subheader("CPI Efficiency Analysis")
        fig = create_cpi_efficiency_chart(won_data, lost_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation
        with st.expander("📊 Interpretation"):
            st.markdown("""
            ### What This Chart Shows
            
            This chart visualizes a combined "efficiency metric" that incorporates IR, LOI, and sample size
            into a single value. Higher efficiency values indicate more favorable project parameters.
            
            ### Key Insights
            
            1. **Efficiency Correlation**: There's a relationship between the efficiency metric and CPI,
               with more efficient projects (higher value) generally having lower CPIs.
            
            2. **Won vs Lost Comparison**: Won bids tend to show better efficiency-to-price ratios than lost bids,
               suggesting that competitive pricing aligned with project parameters is important for winning bids.
            
            3. **Trend Analysis**: The trend lines show how CPI typically scales with efficiency for each bid type,
               helping to identify optimal pricing points based on project parameters.
            """)
        
        # Recent trends section
        st.header("Project Volume Trends")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # Add a date filter if there's date data available
        date_col = None
        for possible_date_col in ['Date', 'Project Date', 'Invoiced Date']:
            if possible_date_col in won_data.columns:
                date_col = possible_date_col
                break
        
        if date_col is not None:
            # Convert date column to datetime if it's not already
            try:
                won_data[date_col] = pd.to_datetime(won_data[date_col])
                
                # Create a date range for the last 12 months
                current_date = won_data[date_col].max()
                one_year_ago = current_date - pd.DateOffset(months=12)
                
                # Filter data for the last 12 months
                recent_won = won_data[won_data[date_col] >= one_year_ago]
                
                # Group by month and count projects
                if not recent_won.empty:
                    recent_won['Month'] = recent_won[date_col].dt.to_period('M')
                    monthly_counts = recent_won.groupby('Month').size().reset_index(name='Count')
                    monthly_counts['Month'] = monthly_counts['Month'].astype(str)
                    
                    # Create bar chart with dark mode styling
                    fig = px.bar(
                        monthly_counts,
                        x='Month',
                        y='Count',
                        title='Monthly Project Volume (Last 12 Months)',
                        labels={'Count': 'Number of Projects', 'Month': ''},
                        color_discrete_sequence=['#4e79a7']
                    )
                    
                    # Update layout for dark mode
                    fig.update_layout(
                        xaxis=dict(
                            tickangle=45,
                            gridcolor='rgba(255,255,255,0.1)',
                            title_font=dict(size=14, color='white')
                        ),
                        yaxis=dict(
                            gridcolor='rgba(255,255,255,0.1)',
                            title_font=dict(size=14, color='white')
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        title=dict(
                            text='Monthly Project Volume (Last 12 Months)',
                            font=dict(size=18, color='white'),
                            x=0.5
                        ),
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No project data available for the last 12 months.")
            except Exception as e:
                logger.error(f"Error creating date trends: {e}")
                st.info("Could not analyze project date trends due to data format issues.")
        else:
            st.info("Date information not available for trend analysis.")
        
        # Add footer with call to action and enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; border-left:5px solid #4e79a7;">
            <h3 style="margin-top:0;">Next Steps</h3>
            <p>Explore the <b>CPI Prediction</b> section to find optimal pricing for new projects based on our machine learning models.</p>
            <p>Or visit the <b>Insights & Recommendations</b> section for strategic pricing advice based on data analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error in overview component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the overview component: {str(e)}")
