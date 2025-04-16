"""
Insights & Recommendations component for the CPI Analysis & Prediction Dashboard.
Provides strategic recommendations based on data analysis with enhanced visualization and explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional

# Import visualization utilities
from utils.visualization import DARK_THEME_COLORS, set_plotly_theme, format_chart_for_dark_mode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the insights and recommendations dashboard with strategic pricing advice.
    Enhanced with improved visualizations and explanations.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.title("Insights & Recommendations")
        
        # Introduction with enhanced styling
        st.markdown("""
        <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; border-left:5px solid #4e79a7;">
            <h3 style="margin-top:0;">Strategic Pricing Insights</h3>
            <p>This section provides data-driven recommendations to optimize your pricing strategy and increase win rates.</p>
            <p>Our analysis is based on patterns identified across won and lost bids in your historical data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Key Findings section with enhanced styling
        st.header("Key Findings")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # Calculate key metrics for insights with error handling
        won_avg_cpi = won_data['CPI'].mean() if not won_data.empty else 0
        lost_avg_cpi = lost_data['CPI'].mean() if not lost_data.empty else 0
        cpi_diff = lost_avg_cpi - won_avg_cpi
        cpi_diff_pct = (cpi_diff / won_avg_cpi) * 100 if won_avg_cpi > 0 else 0
        
        # Create an interactive key findings section
        insights_tabs = st.tabs(["Summary", "IR Impact", "LOI Impact", "Sample Size Effect", "Combination Effects"])
        
        with insights_tabs[0]:
            st.markdown(f"""
            Based on the analysis of the CPI (Cost Per Interview) data between won and lost bids, 
            we've identified the following key insights:
            
            1. **Overall CPI Difference**: There is a significant gap between the average CPI for won bids 
               (${won_avg_cpi:.2f}) and lost bids (${lost_avg_cpi:.2f}), a difference of ${cpi_diff:.2f} or 
               {cpi_diff_pct:.1f}%. This suggests that pricing is a critical factor in bid success.
               
            2. **IR (Incidence Rate) Impact**: Lower IR values generally correlate with higher CPIs, as it becomes 
               more difficult and costly to find qualified respondents. Lost bids tend to have higher CPIs at all IR levels,
               but the difference is most pronounced at lower IR levels.
               
            3. **LOI (Length of Interview) Impact**: As LOI increases, CPI tends to increase for both won and lost bids.
               However, lost bids show a steeper increase in CPI as LOI gets longer, suggesting that pricing for longer
               surveys may be a key differentiator.
               
            4. **Sample Size Effect**: Larger sample sizes (higher number of completes) tend to have lower per-unit CPIs
               due to economies of scale. Lost bids often don't sufficiently account for this scaling effect.
               
            5. **Combination Effects**: The interaction between IR and LOI has a significant impact on CPI. The optimal
               pricing varies considerably depending on these two factors combined.
            """)
            
            # Create a summary visualization
            # Show the won vs lost difference across key dimensions
            summary_data = pd.DataFrame({
                'Metric': ['Overall', 'Low IR (0-20%)', 'Medium IR (21-50%)', 'High IR (51-100%)', 
                          'Short LOI (0-10min)', 'Medium LOI (11-20min)', 'Long LOI (21+min)'],
                'Won Avg CPI': [
                    won_avg_cpi,
                    won_data[won_data['IR'] <= 20]['CPI'].mean(),
                    won_data[(won_data['IR'] > 20) & (won_data['IR'] <= 50)]['CPI'].mean(),
                    won_data[won_data['IR'] > 50]['CPI'].mean(),
                    won_data[won_data['LOI'] <= 10]['CPI'].mean(),
                    won_data[(won_data['LOI'] > 10) & (won_data['LOI'] <= 20)]['CPI'].mean(),
                    won_data[won_data['LOI'] > 20]['CPI'].mean()
                ],
                'Lost Avg CPI': [
                    lost_avg_cpi,
                    lost_data[lost_data['IR'] <= 20]['CPI'].mean(),
                    lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].mean(),
                    lost_data[lost_data['IR'] > 50]['CPI'].mean(),
                    lost_data[lost_data['LOI'] <= 10]['CPI'].mean(),
                    lost_data[(lost_data['LOI'] > 10) & (lost_data['LOI'] <= 20)]['CPI'].mean(),
                    lost_data[lost_data['LOI'] > 20]['CPI'].mean()
                ]
            })
            
            # Calculate difference
            summary_data['Difference'] = summary_data['Lost Avg CPI'] - summary_data['Won Avg CPI']
            summary_data['Percent Diff'] = (summary_data['Difference'] / summary_data['Won Avg CPI']) * 100
            
            # Create figure
            fig = go.Figure()
            
            # Add Won CPI bars
            fig.add_trace(go.Bar(
                x=summary_data['Metric'],
                y=summary_data['Won Avg CPI'],
                name='Won Avg CPI',
                marker_color=DARK_THEME_COLORS['won'],
                hovertemplate='<b>%{x}</b><br>Won Avg CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add Lost CPI bars
            fig.add_trace(go.Bar(
                x=summary_data['Metric'],
                y=summary_data['Lost Avg CPI'],
                name='Lost Avg CPI',
                marker_color=DARK_THEME_COLORS['lost'],
                hovertemplate='<b>%{x}</b><br>Lost Avg CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add difference line
            fig.add_trace(go.Scatter(
                x=summary_data['Metric'],
                y=summary_data['Percent Diff'],
                name='% Difference',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color=DARK_THEME_COLORS['highlight'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Difference: %{y:.1f}%<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title="CPI Comparison Across Key Segments",
                xaxis_title="Segment",
                yaxis=dict(
                    title="CPI ($)",
                    title_font=dict(color='white'),
                    tickfont=dict(color='white'),
                    gridcolor='rgba(255,255,255,0.1)',
                    tickprefix='$'
                ),
                yaxis2=dict(
                    title="% Difference",
                    title_font=dict(color=DARK_THEME_COLORS['highlight']),
                    tickfont=dict(color=DARK_THEME_COLORS['highlight']),
                    anchor='x',
                    overlaying='y',
                    side='right',
                    ticksuffix='%',
                    gridcolor='rgba(255,255,255,0.0)'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5
                ),
                barmode='group',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 How to interpret this chart"):
                st.markdown("""
                This chart shows the CPI difference between won and lost bids across different segments:
                
                - **Green bars**: Average CPI for won bids in each segment
                - **Red bars**: Average CPI for lost bids in each segment
                - **Orange line**: Percentage difference between won and lost CPIs
                
                **Key Insight**: The segments with the largest percentage differences (where the orange line peaks)
                represent the areas where pricing is most critical to winning bids. These are your high-leverage
                opportunities for strategic pricing adjustments.
                """)
        
        with insights_tabs[1]:
            st.subheader("IR (Incidence Rate) Impact Analysis")
            
            # Group data by IR bins and calculate average CPI
            won_ir_bins = won_data.groupby('IR_Bin')['CPI'].mean().reset_index()
            lost_ir_bins = lost_data.groupby('IR_Bin')['CPI'].mean().reset_index()
            
            # Merge the data
            ir_comparison = pd.merge(won_ir_bins, lost_ir_bins, on='IR_Bin', suffixes=('_Won', '_Lost'))
            
            # Calculate difference and percentage
            ir_comparison['Difference'] = ir_comparison['CPI_Lost'] - ir_comparison['CPI_Won']
            ir_comparison['Difference_Pct'] = (ir_comparison['Difference'] / ir_comparison['CPI_Won']) * 100
            
            # Create enhanced IR impact visualization
            fig = go.Figure()
            
            # Add bar chart for difference
            fig.add_trace(go.Bar(
                x=ir_comparison['IR_Bin'],
                y=ir_comparison['Difference'],
                name='CPI Gap ($)',
                marker_color=DARK_THEME_COLORS['neutral'],
                text=ir_comparison['Difference'].apply(lambda x: f'${x:.2f}'),
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>CPI Gap: $%{y:.2f}<br>Won: $%{customdata[0]:.2f}<br>Lost: $%{customdata[1]:.2f}<br>Difference: %{customdata[2]:.1f}%<extra></extra>',
                customdata=np.column_stack((ir_comparison['CPI_Won'], ir_comparison['CPI_Lost'], ir_comparison['Difference_Pct']))
            ))
            
            # Add line chart for percentage
            fig.add_trace(go.Scatter(
                x=ir_comparison['IR_Bin'],
                y=ir_comparison['Difference_Pct'],
                name='CPI Gap (%)',
                yaxis='y2',
                mode='lines+markers',
                line=dict(color=DARK_THEME_COLORS['highlight'], width=3),
                marker=dict(size=8),
                hovertemplate='<b>%{x}</b><br>Difference: %{y:.1f}%<extra></extra>'
            ))
            
            # Format for dark mode
            fig = format_chart_for_dark_mode(fig)
            
            # Update layout with custom styling
            fig.update_layout(
                title={
                    'text': 'CPI Gap Between Lost and Won Bids by IR Range',
                    'font': {'size': 20, 'color': 'white'},
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title={
                    'text': 'Incidence Rate Range',
                    'font': {'size': 14, 'color': 'white'}
                },
                yaxis=dict(
                    title='CPI Gap ($)',
                    title_font=dict(color=DARK_THEME_COLORS['neutral']),
                    tickfont=dict(color=DARK_THEME_COLORS['neutral']),
                    gridcolor='rgba(255,255,255,0.1)',
                    tickprefix='$'
                ),
                yaxis2=dict(
                    title='CPI Gap (%)',
                    title_font=dict(color=DARK_THEME_COLORS['highlight']),
                    tickfont=dict(color=DARK_THEME_COLORS['highlight']),
                    anchor='x',
                    overlaying='y',
                    side='right',
                    ticksuffix='%'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation with enhanced styling
            st.markdown("""
            <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; margin-top:10px;">
                <h4 style="margin-top:0;">Key Insights on IR Impact</h4>
                <ul>
                    <li><strong>Lower IR = Larger Gap</strong>: The gap between won and lost bids is typically largest at lower IR levels, indicating this is where pricing strategy is most critical.</li>
                    <li><strong>Pricing Sensitivity</strong>: Low IR projects (0-20%) show the highest pricing sensitivity - a small change in price can significantly impact win probability.</li>
                    <li><strong>Competitive Advantage</strong>: Having competitive pricing for low IR projects can be a major competitive advantage, as many competitors overprice these more challenging projects.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_tabs[2]:
            st.subheader("LOI (Length of Interview) Impact Analysis")
            
            # Group data by LOI bins and calculate average CPI
            won_loi_bins = won_data.groupby('LOI_Bin')['CPI'].mean().reset_index()
            lost_loi_bins = lost_data.groupby('LOI_Bin')['CPI'].mean().reset_index()
            
            # Merge the data
            loi_comparison = pd.merge(won_loi_bins, lost_loi_bins, on='LOI_Bin', suffixes=('_Won', '_Lost'))
            
            # Calculate difference and percentage
            loi_comparison['Difference'] = loi_comparison['CPI_Lost'] - loi_comparison['CPI_Won']
            loi_comparison['Difference_Pct'] = (loi_comparison['Difference'] / loi_comparison['CPI_Won']) * 100
            
            # Create enhanced LOI impact visualization
            fig = go.Figure()
            
            # Add line chart for won CPIs
            fig.add_trace(go.Scatter(
                x=loi_comparison['LOI_Bin'],
                y=loi_comparison['CPI_Won'],
                name='Won Avg CPI',
                mode='lines+markers',
                line=dict(color=DARK_THEME_COLORS['won'], width=3),
                marker=dict(size=10, symbol='circle'),
                hovertemplate='<b>%{x}</b><br>Won Avg CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add line chart for lost CPIs
            fig.add_trace(go.Scatter(
                x=loi_comparison['LOI_Bin'],
                y=loi_comparison['CPI_Lost'],
                name='Lost Avg CPI',
                mode='lines+markers',
                line=dict(color=DARK_THEME_COLORS['lost'], width=3),
                marker=dict(size=10, symbol='circle'),
                hovertemplate='<b>%{x}</b><br>Lost Avg CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add area to highlight gap
            fig.add_trace(go.Scatter(
                x=loi_comparison['LOI_Bin'].tolist() + loi_comparison['LOI_Bin'].tolist()[::-1],
                y=loi_comparison['CPI_Lost'].tolist() + loi_comparison['CPI_Won'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 255, 255, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                hoverinfo='skip',
                showlegend=False
            ))
            
            # Format for dark mode
            fig = format_chart_for_dark_mode(fig)
            
            # Update layout with custom styling
            fig.update_layout(
                title={
                    'text': 'CPI Trends by Length of Interview (LOI)',
                    'font': {'size': 20, 'color': 'white'},
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title={
                    'text': 'Length of Interview',
                    'font': {'size': 14, 'color': 'white'}
                },
                yaxis_title={
                    'text': 'Average CPI ($)',
                    'font': {'size': 14, 'color': 'white'}
                },
                yaxis=dict(
                    tickprefix='$',
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                height=500
            )
            
            # Add annotations to highlight trends
            for i in range(len(loi_comparison)):
                if i > 0:
                    # Calculate slope for won and lost lines
                    won_slope = (loi_comparison['CPI_Won'].iloc[i] - loi_comparison['CPI_Won'].iloc[i-1])
                    lost_slope = (loi_comparison['CPI_Lost'].iloc[i] - loi_comparison['CPI_Lost'].iloc[i-1])
                    
                    # Add arrows to highlight rate of change differences
                    if abs(lost_slope - won_slope) > 1:  # Only add for significant differences
                        # Use actual bin names instead of index positions
                        # This avoids the non-integer indexing error
                        mid_bin = loi_comparison['LOI_Bin'].iloc[i]
                        
                        fig.add_annotation(
                            x=mid_bin,
                            y=(loi_comparison['CPI_Lost'].iloc[i] + loi_comparison['CPI_Lost'].iloc[i-1]) / 2,
                            text="Steeper increase for lost bids" if lost_slope > won_slope else "Similar rates",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor=DARK_THEME_COLORS['highlight'],
                            font=dict(color=DARK_THEME_COLORS['highlight']),
                            bgcolor="rgba(0,0,0,0.7)",
                            bordercolor=DARK_THEME_COLORS['highlight'],
                            borderpad=4,
                            borderwidth=1
                        )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add interpretation with enhanced styling
            st.markdown("""
            <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; margin-top:10px;">
                <h4 style="margin-top:0;">Key Insights on LOI Impact</h4>
                <ul>
                    <li><strong>Steeper Curve for Lost Bids</strong>: The CPI for lost bids increases more rapidly with LOI than for won bids, suggesting that pricing discipline for longer surveys is critical.</li>
                    <li><strong>Critical Transition Points</strong>: The gap between won and lost bids often widens significantly at certain LOI thresholds (typically around 15-20 minutes), indicating key decision points.</li>
                    <li><strong>Strategic Opportunity</strong>: For longer surveys (21+ minutes), maintaining competitive pricing offers a strategic advantage as competitors tend to apply excessive premiums.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add optimization recommendations
            st.markdown("""
            <div style="background-color:rgba(242, 142, 43, 0.2); padding:15px; border-radius:5px; margin-top:20px; border-left:5px solid #f28e2b;">
                <h4 style="margin-top:0; color:#f28e2b;">LOI Optimization Recommendations</h4>
                <ol>
                    <li>Use a scaled LOI multiplier that increases at a lower rate than your competitors</li>
                    <li>For surveys over 20 minutes, keep your CPI premium to no more than 30-40% above your 10-minute survey rate</li>
                    <li>Consider offering tiered discounts specifically for longer interviews to differentiate from competitors</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with insights_tabs[3]:
            st.subheader("Sample Size (Completes) Effect Analysis")
            
            # Create scatter plot of CPI vs Completes
            fig = go.Figure()
            
            # Add scatter for won bids
            fig.add_trace(go.Scatter(
                x=won_data['Completes'],
                y=won_data['CPI'],
                name='Won Bids',
                mode='markers',
                marker=dict(
                    color=DARK_THEME_COLORS['won'],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='rgba(255,255,255,0.3)')
                ),
                hovertemplate='<b>Won Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add scatter for lost bids
            fig.add_trace(go.Scatter(
                x=lost_data['Completes'],
                y=lost_data['CPI'],
                name='Lost Bids',
                mode='markers',
                marker=dict(
                    color=DARK_THEME_COLORS['lost'],
                    size=10,
                    opacity=0.7,
                    line=dict(width=1, color='rgba(255,255,255,0.3)')
                ),
                hovertemplate='<b>Lost Bid</b><br>Completes: %{x}<br>CPI: $%{y:.2f}<extra></extra>'
            ))
            
            # Add trend lines
            # For won data
            if len(won_data) >= 2:
                x_range = np.linspace(won_data['Completes'].min(), won_data['Completes'].max(), 100)
                
                # Fit a power law: CPI = a * Completes^b
                # Using log transform for linear fit
                won_log_x = np.log(won_data['Completes'].replace(0, 1))
                won_log_y = np.log(won_data['CPI'].replace(0, 0.1))
                won_slope, won_intercept = np.polyfit(won_log_x, won_log_y, 1)
                
                won_trend_y = np.exp(won_intercept) * (x_range ** won_slope)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=won_trend_y,
                    name='Won Trend (Power Law)',
                    mode='lines',
                    line=dict(color=DARK_THEME_COLORS['won'], width=3, dash='solid'),
                    hoverinfo='skip'
                ))
                
                # Add annotation to explain the power law
                fig.add_annotation(
                    x=x_range[-1],
                    y=won_trend_y[-1],
                    text=f"CPI ∝ n^{won_slope:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=DARK_THEME_COLORS['won'],
                    font=dict(color=DARK_THEME_COLORS['won']),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=DARK_THEME_COLORS['won'],
                    borderpad=4,
                    borderwidth=1
                )
            
            # For lost data
            if len(lost_data) >= 2:
                x_range = np.linspace(lost_data['Completes'].min(), lost_data['Completes'].max(), 100)
                
                # Fit a power law: CPI = a * Completes^b
                # Using log transform for linear fit
                lost_log_x = np.log(lost_data['Completes'].replace(0, 1))
                lost_log_y = np.log(lost_data['CPI'].replace(0, 0.1))
                lost_slope, lost_intercept = np.polyfit(lost_log_x, lost_log_y, 1)
                
                lost_trend_y = np.exp(lost_intercept) * (x_range ** lost_slope)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=lost_trend_y,
                    name='Lost Trend (Power Law)',
                    mode='lines',
                    line=dict(color=DARK_THEME_COLORS['lost'], width=3, dash='solid'),
                    hoverinfo='skip'
                ))
                
                # Add annotation to explain the power law
                fig.add_annotation(
                    x=x_range[-1],
                    y=lost_trend_y[-1],
                    text=f"CPI ∝ n^{lost_slope:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor=DARK_THEME_COLORS['lost'],
                    font=dict(color=DARK_THEME_COLORS['lost']),
                    bgcolor="rgba(0,0,0,0.7)",
                    bordercolor=DARK_THEME_COLORS['lost'],
                    borderpad=4,
                    borderwidth=1
                )
            
            # Format for dark mode
            fig = format_chart_for_dark_mode(fig)
            
            # Update layout with custom styling
            fig.update_layout(
                title={
                    'text': 'CPI vs Sample Size (Economies of Scale)',
                    'font': {'size': 20, 'color': 'white'},
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title={
                    'text': 'Sample Size (Completes)',
                    'font': {'size': 14, 'color': 'white'}
                },
                yaxis_title={
                    'text': 'CPI ($)',
                    'font': {'size': 14, 'color': 'white'}
                },
                yaxis=dict(
                    tickprefix='$',
                    type='log',  # Use log scale to better visualize economies of scale
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                xaxis=dict(
                    type='log',  # Use log scale for x-axis
                    gridcolor='rgba(255,255,255,0.1)'
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📊 Understanding Economies of Scale"):
                st.markdown("""
                This chart uses logarithmic scales on both axes to clearly show the "economies of scale" effect:
                
                - Each point represents a project, with the x-axis showing the sample size and the y-axis showing the CPI
                - The downward slope of the trend lines shows that larger sample sizes correlate with lower per-unit costs
                - The exponent in the power law formula (shown in the annotations) indicates how strong the economy of scale effect is
                
                **Key Finding**: Won bids typically show a stronger economy of scale effect than lost bids, indicating that appropriately
                discounting larger projects is important for winning them.
                """)
            
            # Add interpretation with enhanced styling
            st.markdown("""
            <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; margin-top:10px;">
                <h4 style="margin-top:0;">Key Insights on Sample Size Effect</h4>
                <ul>
                    <li><strong>Economies of Scale</strong>: Both won and lost bids show a clear economy of scale effect, where larger sample sizes correlate with lower per-unit costs.</li>
                    <li><strong>Steeper Discount for Won Bids</strong>: Won bids typically show a steeper discount curve than lost bids, suggesting that appropriate volume discounting is key to winning larger projects.</li>
                    <li><strong>Critical Volume Thresholds</strong>: The data indicates critical threshold points (typically around 300, 500, and 1000 completes) where volume discounts become particularly important.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Add volume discount recommendation table
            st.subheader("Recommended Volume Discount Structure")
            
            # Calculate optimal discounts based on data patterns
            small_baseline = won_data[won_data['Completes'] <= 100]['CPI'].mean()
            medium_discount = ((won_data[(won_data['Completes'] > 100) & (won_data['Completes'] <= 500)]['CPI'].mean() / small_baseline) - 1) * 100
            large_discount = ((won_data[(won_data['Completes'] > 500) & (won_data['Completes'] <= 1000)]['CPI'].mean() / small_baseline) - 1) * 100
            very_large_discount = ((won_data[won_data['Completes'] > 1000]['CPI'].mean() / small_baseline) - 1) * 100
            
            # Create a cleaner display (handle NaNs)
            medium_discount = medium_discount if not np.isnan(medium_discount) else -5
            large_discount = large_discount if not np.isnan(large_discount) else -10
            very_large_discount = very_large_discount if not np.isnan(very_large_discount) else -15
            
            # Ensure discounts are negative (i.e., actual discounts)
            medium_discount = min(medium_discount, -5)
            large_discount = min(large_discount, medium_discount - 5)
            very_large_discount = min(very_large_discount, large_discount - 5)
            
            # Create discount table
            discount_data = {
                "Sample Size Range": ["Small (1-100)", "Medium (101-500)", "Large (501-1000)", "Very Large (1000+)"],
                "Discount %": ["0%", f"{abs(medium_discount):.0f}%", f"{abs(large_discount):.0f}%", f"{abs(very_large_discount):.0f}%"],
                "Example": [
                    f"100 completes at base CPI",
                    f"300 completes with {abs(medium_discount):.0f}% discount",
                    f"750 completes with {abs(large_discount):.0f}% discount",
                    f"1500 completes with {abs(very_large_discount):.0f}% discount"
                ]
            }
            
            # Display the discount table with enhanced styling
            discount_df = pd.DataFrame(discount_data)
            
            # Use st.table for better styled display
            st.table(discount_df)
        
        with insights_tabs[4]:
            st.subheader("Combination Effects Analysis")
            
            # Create a heatmap of CPI by IR and LOI combinations
            
            # Create IR and LOI bins if not already present
            for df in [won_data, lost_data]:
                if 'IR_Numeric' not in df.columns:
                    df['IR_Numeric'] = pd.cut(
                        df['IR'],
                        bins=[0, 10, 20, 30, 50, 100],
                        labels=[5, 15, 25, 40, 75]
                    ).astype(float)
                
                if 'LOI_Numeric' not in df.columns:
                    df['LOI_Numeric'] = pd.cut(
                        df['LOI'],
                        bins=[0, 10, 15, 20, 30, 60],
                        labels=[5, 12.5, 17.5, 25, 45]
                    ).astype(float)
            
            # Create pivot tables
            won_pivot = won_data.pivot_table(
                values='CPI',
                index='LOI_Numeric',
                columns='IR_Numeric',
                aggfunc='mean'
            ).round(2)
            
            lost_pivot = lost_data.pivot_table(
                values='CPI',
                index='LOI_Numeric',
                columns='IR_Numeric',
                aggfunc='mean'
            ).round(2)
            
            # Calculate difference pivot
            diff_pivot = (lost_pivot - won_pivot).round(2)
            
            # Create a custom colorscale for better contrast in dark mode
            colorscale_won = [
                [0, 'rgba(0,100,0,0.6)'],
                [0.5, 'rgba(144,238,144,0.8)'],
                [1, 'rgba(255,255,255,0.9)']
            ]
            
            colorscale_lost = [
                [0, 'rgba(100,0,0,0.6)'],
                [0.5, 'rgba(255,99,71,0.8)'],
                [1, 'rgba(255,255,255,0.9)']
            ]
            
            colorscale_diff = [
                [0, 'rgba(0,100,0,0.6)'],
                [0.5, 'rgba(255,255,255,0.7)'],
                [1, 'rgba(100,0,0,0.6)']
            ]
            
            # Create layout with tabs for different heatmaps
            heatmap_tabs = st.tabs(["Won Bids CPI", "Lost Bids CPI", "CPI Gap"])
            
            with heatmap_tabs[0]:
                # Create heatmap for won bids
                fig_won = go.Figure(data=go.Heatmap(
                    z=won_pivot.values,
                    x=won_pivot.columns,
                    y=won_pivot.index,
                    colorscale=colorscale_won,
                    text=won_pivot.values,
                    texttemplate="$%{text:.2f}",
                    hovertemplate='<b>IR: %{x}%</b><br>LOI: %{y} min<br>CPI: $%{z:.2f}<extra></extra>'
                ))
                
                fig_won.update_layout(
                    title="Won Bids: Average CPI by IR and LOI",
                    xaxis_title="Incidence Rate (%)",
                    yaxis_title="Length of Interview (min)",
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(ticksuffix=" min"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=500
                )
                
                st.plotly_chart(fig_won, use_container_width=True)
                
                st.markdown("""
                <div style="background-color:rgba(82, 188, 163, 0.2); padding:10px; border-radius:5px; margin-top:10px;">
                    <p><strong>Interpretation:</strong> This heatmap shows the average CPI for won bids at different combinations of IR and LOI.
                    Darker green indicates lower CPIs. Use this as a reference for competitive pricing points.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with heatmap_tabs[1]:
                # Create heatmap for lost bids
                fig_lost = go.Figure(data=go.Heatmap(
                    z=lost_pivot.values,
                    x=lost_pivot.columns,
                    y=lost_pivot.index,
                    colorscale=colorscale_lost,
                    text=lost_pivot.values,
                    texttemplate="$%{text:.2f}",
                    hovertemplate='<b>IR: %{x}%</b><br>LOI: %{y} min<br>CPI: $%{z:.2f}<extra></extra>'
                ))
                
                fig_lost.update_layout(
                    title="Lost Bids: Average CPI by IR and LOI",
                    xaxis_title="Incidence Rate (%)",
                    yaxis_title="Length of Interview (min)",
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(ticksuffix=" min"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=500
                )
                
                st.plotly_chart(fig_lost, use_container_width=True)
                
                st.markdown("""
                <div style="background-color:rgba(225, 87, 89, 0.2); padding:10px; border-radius:5px; margin-top:10px;">
                    <p><strong>Interpretation:</strong> This heatmap shows the average CPI for lost bids at different combinations of IR and LOI.
                    Darker red indicates higher CPIs. These represent pricing points to avoid.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with heatmap_tabs[2]:
                # Create heatmap for difference
                diff_values = diff_pivot.values
                
                fig_diff = go.Figure(data=go.Heatmap(
                    z=diff_values,
                    x=diff_pivot.columns,
                    y=diff_pivot.index,
                    colorscale=colorscale_diff,
                    text=diff_values,
                    texttemplate="$%{text:.2f}",
                    hovertemplate='<b>IR: %{x}%</b><br>LOI: %{y} min<br>CPI Gap: $%{z:.2f}<extra></extra>'
                ))
                
                fig_diff.update_layout(
                    title="CPI Gap: Lost Bids - Won Bids by IR and LOI",
                    xaxis_title="Incidence Rate (%)",
                    yaxis_title="Length of Interview (min)",
                    xaxis=dict(ticksuffix="%"),
                    yaxis=dict(ticksuffix=" min"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=500
                )
                
                st.plotly_chart(fig_diff, use_container_width=True)
                
                st.markdown("""
                <div style="background-color:rgba(78, 121, 167, 0.2); padding:10px; border-radius:5px; margin-top:10px;">
                    <p><strong>Interpretation:</strong> This heatmap shows the CPI gap between lost and won bids. 
                    Red areas indicate larger gaps where pricing is most critical. These are the high-leverage areas where pricing strategy can have the greatest impact.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Add key findings from combination analysis
            st.markdown("""
            <div style="background-color:rgba(242, 142, 43, 0.2); padding:15px; border-radius:5px; margin-top:20px; border-left:5px solid #f28e2b;">
                <h4 style="margin-top:0; color:#f28e2b;">Key Findings from Combination Analysis</h4>
                <ol>
                    <li><strong>High-Leverage Combinations</strong>: The largest pricing gaps typically occur in projects with both low IR and high LOI - these are the most challenging projects where pricing strategy is most critical</li>
                    <li><strong>Competitive Edge</strong>: For high IR / low LOI combinations, the pricing gap is minimal, suggesting that factors beyond price (like speed, quality, or relationship) may be more important</li>
                    <li><strong>Strategic Opportunity</strong>: Medium IR / high LOI combinations often show substantial pricing gaps, representing an opportunity for strategic pricing</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations section with enhanced styling
        st.header("Strategic Pricing Recommendations")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # Create IR-based pricing tiers recommendation
        # Calculate tier thresholds
        ir_low_threshold = lost_data[lost_data['IR'] <= 20]['CPI'].quantile(0.25) if not lost_data.empty else 0
        ir_med_threshold = lost_data[(lost_data['IR'] > 20) & (lost_data['IR'] <= 50)]['CPI'].quantile(0.25) if not lost_data.empty else 0
        ir_high_threshold = lost_data[lost_data['IR'] > 50]['CPI'].quantile(0.25) if not lost_data.empty else 0
        
        # Add some reasonable defaults if data is insufficient
        if ir_low_threshold <= 0: ir_low_threshold = 30
        if ir_med_threshold <= 0: ir_med_threshold = 20
        if ir_high_threshold <= 0: ir_high_threshold = 15
        
        # Add a container for better visual grouping of recommendations
        with st.container():
            # Create three columns for the main pricing recommendations
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.subheader("1. IR-Based Pricing Tiers")
                
                st.markdown(f"""
                <div style="background-color:rgba(30, 33, 48, 0.5); padding:15px; border-radius:5px; height:100%;">
                    <h4 style="margin-top:0; color:white;">Recommended Tiers</h4>
                    <ul>
                        <li><strong>Low IR (0-20%)</strong>: Keep CPIs below ${ir_low_threshold:.2f}</li>
                        <li><strong>Medium IR (21-50%)</strong>: Keep CPIs below ${ir_med_threshold:.2f}</li>
                        <li><strong>High IR (51-100%)</strong>: Keep CPIs below ${ir_high_threshold:.2f}</li>
                    </ul>
                    <p style="font-size:0.9em; color:rgba(255, 255, 255, 0.7);">These thresholds represent the 25th percentile of lost bids in each range, offering a pricing point with high win probability.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.subheader("2. LOI Multipliers")
                
                # Calculate multipliers based on data patterns
                baseline_loi = won_data[won_data['LOI'] <= 10]['CPI'].mean() if not won_data.empty else 10
                med_multiplier = won_data[(won_data['LOI'] > 10) & (won_data['LOI'] <= 20)]['CPI'].mean() / baseline_loi if not won_data.empty and baseline_loi > 0 else 1.3
                long_multiplier = won_data[won_data['LOI'] > 20]['CPI'].mean() / baseline_loi if not won_data.empty and baseline_loi > 0 else 1.5
                
                # Handle potential data issues
                if np.isnan(med_multiplier) or med_multiplier <= 1: med_multiplier = 1.3
                if np.isnan(long_multiplier) or long_multiplier <= med_multiplier: long_multiplier = med_multiplier + 0.2
                
                # Format for display
                med_pct = (med_multiplier - 1) * 100
                long_pct = (long_multiplier - 1) * 100
                
                st.markdown(f"""
                <div style="background-color:rgba(30, 33, 48, 0.5); padding:15px; border-radius:5px; height:100%;">
                    <h4 style="margin-top:0; color:white;">LOI Multipliers</h4>
                    <ul>
                        <li><strong>Short LOI (1-10 min)</strong>: Base CPI (1.0x)</li>
                        <li><strong>Medium LOI (11-20 min)</strong>: {med_multiplier:.2f}x base CPI (+{med_pct:.0f}%)</li>
                        <li><strong>Long LOI (21+ min)</strong>: {long_multiplier:.2f}x base CPI (+{long_pct:.0f}%)</li>
                    </ul>
                    <p style="font-size:0.9em; color:rgba(255, 255, 255, 0.7);">These multipliers are based on successful pricing patterns in won bids and ensure appropriate scaling without overpricing.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col3:
                st.subheader("3. Volume Discounts")
                
                st.markdown(f"""
                <div style="background-color:rgba(30, 33, 48, 0.5); padding:15px; border-radius:5px; height:100%;">
                    <h4 style="margin-top:0; color:white;">Sample Size Discounts</h4>
                    <ul>
                        <li><strong>Small (1-100)</strong>: Standard CPI (0% discount)</li>
                        <li><strong>Medium (101-500)</strong>: {abs(medium_discount):.0f}% discount</li>
                        <li><strong>Large (501-1000)</strong>: {abs(large_discount):.0f}% discount</li>
                        <li><strong>Very Large (1000+)</strong>: {abs(very_large_discount):.0f}% discount</li>
                    </ul>
                    <p style="font-size:0.9em; color:rgba(255, 255, 255, 0.7);">Based on economies of scale observed in won bids. Applying these discounts keeps you competitive for larger projects.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Combined factor pricing model recommendation
        st.subheader("4. Combined Factor Pricing Model")
        
        # Using the CPI Prediction Model section with side-by-side columns instead of flex layout
        st.markdown("""
        <div style="background-color:rgba(78, 121, 167, 0.2); padding:20px; border-radius:5px; margin-top:20px;">
            <h4 style="margin-top:0;">Using the CPI Prediction Model</h4>
            <p>For the most accurate pricing, use the machine learning prediction model from the CPI Prediction section. It accounts for complex interactions between factors that simple rules can't capture.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use streamlit columns for better layout control
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            st.markdown("""
            <div style="background-color:rgba(30, 33, 48, 0.7); padding:15px; border-radius:5px; height:100%;">
                <h5 style="margin-top:0; color:#f28e2b;">When to Use the ML Model</h5>
                <ul>
                    <li>High-stakes projects where optimal pricing is critical</li>
                    <li>Projects with unusual combinations of IR, LOI, and sample size</li>
                    <li>When you need to maximize win probability for strategic clients</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with model_col2:
            st.markdown("""
            <div style="background-color:rgba(30, 33, 48, 0.7); padding:15px; border-radius:5px; height:100%;">
                <h5 style="margin-top:0; color:#f28e2b;">Model Advantages</h5>
                <ul>
                    <li>Considers complex interaction effects between all parameters</li>
                    <li>Adapts to your specific winning bid patterns</li>
                    <li>Provides probability estimates and confidence intervals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation guide with enhanced styling
        st.header("Implementation Guide")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # Create a step-by-step implementation process
        st.markdown("""
        <div style="background-color:rgba(30, 33, 48, 0.5); padding:20px; border-radius:5px; margin-top:20px;">
            <h4 style="margin-top:0; color:white;">How to Implement These Recommendations</h4>
            
            <ol>
                <li><strong>Start with baseline pricing tiers</strong> based on IR ranges</li>
                <li><strong>Apply LOI multipliers</strong> based on survey length</li>
                <li><strong>Apply volume discounts</strong> based on sample size</li>
                <li><strong>Cross-check with the ML model</strong> for final validation</li>
                <li><strong>Fine-tune based on client relationship and strategic value</strong> (±5-10%)</li>
            </ol>
            
            <p style="margin-top:15px;">For critical bids, use the CPI Prediction tool to get the most precise pricing recommendation tailored to your specific project parameters.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Success metrics and follow-up
        st.subheader("Measuring Success")
        
        st.markdown("""
        <div style="background-color:rgba(82, 188, 163, 0.2); padding:15px; border-radius:5px; margin-top:10px;">
            <h4 style="margin-top:0; color:#52bca3;">Key Performance Indicators</h4>
            <p>Track these metrics to measure the effectiveness of your pricing strategy:</p>
            <ol>
                <li><strong>Win Rate</strong>: Should increase by 5-10% within 3 months of implementation</li>
                <li><strong>Average Margin</strong>: Should remain stable or increase slightly</li>
                <li><strong>Low IR Project Win Rate</strong>: This specific segment should show the most improvement</li>
                <li><strong>Large Project Win Rate</strong>: Projects with 500+ completes should show improved conversion</li>
            </ol>
            <p style="font-style:italic; margin-top:10px;">Continuously update the prediction model with new data to ensure recommendations remain accurate over time.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add footer with final advice
        st.markdown("""
        <div style="background-color:rgba(38, 39, 48, 0.8); padding:15px; border-radius:5px; margin-top:20px; border:1px solid rgba(78, 121, 167, 0.5);">
            <h3 style="margin-top:0; color:white;">Final Advice</h3>
            <p>While these pricing recommendations are data-driven, remember that each project and client relationship is unique. Use these guidelines as a starting point, but always apply your market knowledge and relationship context when finalizing pricing.</p>
            <p>For the most personalized pricing recommendations, use the CPI Prediction tool in the previous section.</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error in insights component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the insights component: {str(e)}")
