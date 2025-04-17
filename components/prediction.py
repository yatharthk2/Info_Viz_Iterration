"""
CPI Prediction component for the CPI Analysis & Prediction Dashboard.
Provides an interactive tool for predicting optimal CPI with enhanced visualization and explainability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

# Import ML model utilities
from models.trainer import build_models
from models.predictor import (
    predict_cpi, 
    get_recommendation, 
    get_detailed_pricing_strategy,
    simulate_win_probability
)

# Import visualization utilities
from utils.visualization import (
    create_feature_importance_chart,
    create_prediction_comparison_chart,
    DARK_THEME_COLORS
)

# Import explanation utilities
from utils.explanation import (
    explain_feature_importance,
    explain_prediction_factors,
    create_prediction_comparison_with_explanation,
    explain_win_probability,
    create_price_sensitivity_curve
)

# Import data processing utilities
from utils.data_processor import prepare_model_data

# Import new model metrics utilities
from utils.model_metrics import (
    evaluate_model_performance,
    plot_learning_curve,
    create_residuals_plot,
    create_feature_dependence_plot,
    create_model_comparison_chart
)
from utils.theme import COLOR_SYSTEM, format_chart_for_dark_mode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show_prediction(combined_data_engineered: pd.DataFrame, won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the CPI prediction tool for estimating optimal pricing with enhanced visualization and explainability.
    
    Args:
        combined_data_engineered (pd.DataFrame): Engineered DataFrame with features for modeling
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    """
    try:
        st.title("CPI Prediction Model")
        
        # Introduction with enhanced styling
        st.markdown("""
        <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; border-left:5px solid #4e79a7;">
            <h3 style="margin-top:0;">About This Tool</h3>
            <p>This tool uses machine learning to predict the optimal Cost Per Interview (CPI) based on your project parameters.</p>
            <p>Enter your values below to get a prediction and pricing recommendation.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Build models with informative progress message
        with st.spinner("Training prediction models..."):
            with st.expander("🧠 Understanding the Model", expanded=False):
                st.markdown("""
                ### How the Prediction Works

                Our model analyzes historical bid data to understand the relationship between project parameters and pricing outcomes.
                
                The process works like this:
                
                1. **Data Analysis**: We analyze patterns in past won and lost bids
                2. **Feature Engineering**: We transform raw data into meaningful patterns
                3. **Model Training**: Multiple machine learning models are trained and compared
                4. **Ensemble Prediction**: Results from different models are combined for accuracy
                5. **Contextual Recommendation**: The prediction is placed in context of historical bids
                
                The result is a recommended CPI that balances competitiveness with profitability.
                """)
            
            # Check if we have enough data
            if len(combined_data_engineered) < 10:
                st.warning("Not enough data to build reliable prediction models. Please ensure you have sufficient data.")
                return
            
            # Prepare model data
            X, y = prepare_model_data(combined_data_engineered)
            
            # Check if preparation was successful
            if len(X) == 0 or len(y) == 0:
                st.warning("Failed to prepare model data. Please check your dataset for missing values or data format issues.")
                return
            
            # Split data for model evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Build models with option for advanced tuning
            do_tuning = st.sidebar.checkbox("Use advanced model tuning (slower but more accurate)", value=False)
            trained_models, model_scores, feature_importance = build_models(X, y, do_tuning)
            
            # Store split data for advanced model evaluation
            model_evaluation_data = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        
        # Toggle for advanced model details
        show_advanced = st.sidebar.checkbox("Show advanced model details", value=False)
        
        if show_advanced:
            # Display model metrics in expandable section
            with st.expander("🔍 Model Performance Details", expanded=False):
                # Create tabs for different model metrics views
                metric_tabs = st.tabs(["Model Accuracy", "Detailed Metrics", "Model Diagnostics", "Learning Curve"])
                
                with metric_tabs[0]:
                    # Create a sorted list of models by R² score
                    sorted_models = sorted(
                        [(name, scores.get('R²', 0)) for name, scores in model_scores.items()],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Display as a horizontal bar chart
                    if sorted_models:
                        model_names = [m[0] for m in sorted_models]
                        r2_scores = [m[1] for m in sorted_models]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            y=model_names,
                            x=r2_scores,
                            orientation='h',
                            marker_color=[f'rgba(78, 121, 167, {max(0.4, score)})' for score in r2_scores],
                            text=[f'{score:.4f}' for score in r2_scores],
                            textposition='auto',
                            hovertemplate='<b>%{y}</b><br>R² Score: %{x:.4f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title="Model R² Scores (Higher is Better)",
                            xaxis_title="R² Score",
                            yaxis=dict(autorange="reversed"),
                            height=300,
                            margin=dict(l=0, r=0, t=30, b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("R² scores measure how well the model explains the variation in CPI. A score of 1.0 would be perfect prediction.")
                    else:
                        st.warning("Model performance metrics not available")
                
                with metric_tabs[1]:
                    # Create a more detailed table of metrics
                    metrics_display = []
                    for model_name, metrics in model_scores.items():
                        # Create a new dict with the model name and all metrics
                        metrics_row = {'Model': model_name}
                        # Use items to properly convert keys and values
                        for k, v in metrics.items():
                            metrics_row[k] = v
                        metrics_display.append(metrics_row)
                    
                    if metrics_display:
                        st.dataframe(
                            pd.DataFrame(metrics_display).set_index('Model'),
                            use_container_width=True
                        )
                    else:
                        st.warning("Detailed metrics not available")
                        
                with metric_tabs[2]:
                    st.subheader("Model Diagnostics")
                    
                    # Get the best model based on R² score
                    best_model_name = None
                    best_r2 = -float('inf')
                    for model_name, scores in model_scores.items():
                        if 'R²' in scores and scores['R²'] > best_r2:
                            best_r2 = scores['R²']
                            best_model_name = model_name
                    
                    if best_model_name and best_model_name in trained_models:
                        best_model = trained_models[best_model_name]
                        
                        # Evaluate model performance
                        evaluation = evaluate_model_performance(
                            best_model, 
                            model_evaluation_data['X_train'], 
                            model_evaluation_data['X_test'],
                            model_evaluation_data['y_train'], 
                            model_evaluation_data['y_test']
                        )
                        
                        # Display model evaluation
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Model**: {best_model_name}")
                            st.markdown(f"**Fit Assessment**: {evaluation['fitting_assessment']}")
                            
                        with col2:
                            # Visualize train/test metrics
                            metrics_to_show = ['R²', 'RMSE', 'MAE']
                            fig = go.Figure()
                            
                            for i, metric in enumerate(metrics_to_show):
                                train_val = evaluation[metric]['train']
                                test_val = evaluation[metric]['test']
                                
                                fig.add_trace(go.Bar(
                                    x=[f"{metric} (Train)", f"{metric} (Test)"],
                                    y=[train_val, test_val],
                                    name=metric,
                                    marker_color=COLOR_SYSTEM['CHARTS'][f'SERIES{i+1}' if i < 7 else 'SERIES1'],
                                ))
                            
                            fig = format_chart_for_dark_mode(fig, "Train vs Test Metrics", height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show assessment and recommendations
                        st.subheader("Assessment")
                        
                        if evaluation['fitting_assessment'] == 'Overfitting':
                            st.warning(f"⚠️ **Overfitting Detected**: Train R² = {evaluation['R²']['train']:.4f}, Test R² = {evaluation['R²']['test']:.4f}")
                        elif evaluation['fitting_assessment'] == 'Underfitting':
                            st.warning(f"⚠️ **Underfitting Detected**: Train R² = {evaluation['R²']['train']:.4f}, Test R² = {evaluation['R²']['test']:.4f}")
                        else:
                            st.success(f"✅ **Good Model Fit**: Train R² = {evaluation['R²']['train']:.4f}, Test R² = {evaluation['R²']['test']:.4f}")
                        
                        st.markdown(f"**Recommendation**: {evaluation['recommendation']}")
                        
                        # Create residuals plot
                        st.subheader("Residuals Analysis")
                        residuals_fig = create_residuals_plot(
                            best_model, 
                            model_evaluation_data['X_test'], 
                            model_evaluation_data['y_test']
                        )
                        st.plotly_chart(residuals_fig, use_container_width=True)
                        
                        # Add explanation for residuals plot - Using text format instead of expander to avoid nesting
                        st.markdown("""
                        ### 📈 How to interpret residuals
                        
                        **Residuals** are the differences between actual and predicted values.
                        
                        - **Scattered randomly**: Good model fit (desired)
                        - **Pattern/trend**: Model may be missing important factors
                        - **Points far from zero**: Potential outliers or poor predictions
                        - **More points above/below zero**: Model may be biased
                        """)
                    else:
                        st.warning("Model diagnostics not available. Please train models first.")
                        
                with metric_tabs[3]:
                    st.subheader("Learning Curve Analysis")
                    
                    # Get the best model
                    if best_model_name and best_model_name in trained_models:
                        best_model = trained_models[best_model_name]
                        
                        # Generate learning curve plot
                        learning_curve_fig = plot_learning_curve(
                            best_model, 
                            X, 
                            y
                        )
                        st.plotly_chart(learning_curve_fig, use_container_width=True)
                        
                        # Add feature dependence plot for key features
                        st.subheader("Feature Effect on CPI")
                        
                        # Select key features to analyze based on feature importance
                        key_features = []
                        if not feature_importance.empty:
                            # Use pandas dataframe directly - it's already sorted
                            top_features = feature_importance.head(3).values
                            key_features = [str(feat) for feat, _ in top_features]
                        else:
                            # Default to basic features if no feature importance available
                            key_features = ['IR', 'LOI', 'Completes']
                        
                        # Create tabs for each feature
                        feature_tabs = st.tabs(key_features)
                        
                        for i, feature in enumerate(key_features):
                            with feature_tabs[i]:
                                if feature in X.columns:
                                    feature_plot = create_feature_dependence_plot(
                                        best_model, 
                                        X, 
                                        feature
                                    )
                                    st.plotly_chart(feature_plot, use_container_width=True)
                                    
                                    # Add explanation of what this means for pricing
                                    if feature == 'IR':
                                        st.markdown("""
                                        **Pricing Strategy**: For low IR projects, consider a premium 
                                        as they're more expensive to field. High IR projects allow for more 
                                        competitive pricing.
                                        """)
                                    elif feature == 'LOI':
                                        st.markdown("""
                                        **Pricing Strategy**: Longer surveys require higher incentives and 
                                        have higher dropout rates. Consider a higher CPI for longer LOIs.
                                        """)
                                    elif feature == 'Completes':
                                        st.markdown("""
                                        **Pricing Strategy**: Larger sample sizes benefit from economies of scale. 
                                        Consider volume discounts for large projects to remain competitive.
                                        """)
                                else:
                                    st.warning(f"Feature '{feature}' not found in the dataset")
                    else:
                        st.warning("Learning curve analysis not available. Please train models first.")
            
            # Show enhanced feature importance visualization
            st.header("Feature Importance Analysis")
            
            if not feature_importance.empty:
                # Create enhanced feature importance chart
                fig = explain_feature_importance(feature_importance, num_features=10)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add feature explanation table
                st.subheader("Understanding Key Features")
                st.markdown("""
                | Feature | What It Means | Impact on CPI |
                |---------|---------------|---------------|
                | IR | Incidence Rate - % of people who qualify | Higher IR = Lower CPI |
                | LOI | Length of Interview in minutes | Longer LOI = Higher CPI |
                | Completes | Sample size required | More Completes = Economies of scale |
                | IR_LOI_Ratio | Ratio of IR to LOI | Higher ratio = More favorable |
                | IR_Squared | Squared value of IR | Captures non-linear effects |
                | Log_Completes | Log transform of Completes | Captures diminishing returns |
                """)
                
                # Add interpretation
                # Using styled markdown instead of nested expander
                st.markdown("""
                ### 📊 Interpreting Feature Importance
                
                Feature importance shows which factors have the strongest influence on CPI in our model.
                Longer bars indicate more significant impact on the predicted price.
                
                ### Key Insights
                
                1. **Primary Drivers**: The top features have the strongest influence on CPI predictions.
                   These should be your primary focus when estimating prices.
                
                2. **Relative Importance**: The values represent the relative importance of each feature
                   compared to others. For example, a feature with 0.4 importance has twice the influence
                   of a feature with 0.2 importance.
                
                3. **Strategic Focus**: When negotiating or adjusting bids, focus on the top features
                   as they will have the largest impact on competitive pricing.
                """)
            else:
                st.warning("Feature importance analysis is not available. This may be due to the model type or insufficient data.")
        
        # User input section with enhanced styling
        st.header("Predict CPI")
        st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px'>", unsafe_allow_html=True)
        
        # Create a container with custom styling for the input form
        with st.container():
            st.markdown("""
            <div style="background-color:rgba(30, 33, 48, 0.5); padding:15px; border-radius:5px;">
                <h3 style="margin-top:0; color:#ffffff;">Enter Project Parameters</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create 3 columns for inputs
            col1, col2, col3 = st.columns(3)
            
            # Calculate min, max, and default values from data with more robust safeguards to prevent equal min/max
            try:
                # Print debugging information
                logger.info(f"IR min raw: {combined_data_engineered['IR'].min()}, IR max raw: {combined_data_engineered['IR'].max()}")
                logger.info(f"LOI min raw: {combined_data_engineered['LOI'].min()}, LOI max raw: {combined_data_engineered['LOI'].max()}")
                logger.info(f"Completes min raw: {combined_data_engineered['Completes'].min()}, Completes max raw: {combined_data_engineered['Completes'].max()}")
                
                # Incidence Rate (IR) slider
                ir_min = max(1, int(combined_data_engineered['IR'].min()))
                ir_max = min(100, int(combined_data_engineered['IR'].max()))
                # Force min/max to be different
                if ir_min >= ir_max:
                    if ir_max > 11:  # If we can safely subtract
                        ir_min = ir_max - 10
                    else:
                        ir_min = 1
                        ir_max = 20  # Force a sensible range
                ir_default = int((ir_min + ir_max) / 2)
                
                # Length of Interview (LOI) slider
                loi_min = max(1, int(combined_data_engineered['LOI'].min()))
                loi_max = min(60, int(combined_data_engineered['LOI'].max() * 1.2))  # Add some buffer
                # Force min/max to be different
                if loi_min >= loi_max:
                    if loi_max > 6:  # If we can safely subtract
                        loi_min = loi_max - 5
                    else:
                        loi_min = 1
                        loi_max = 15  # Force a sensible range
                loi_default = int((loi_min + loi_max) / 2)
                
                # Completes slider
                completes_min = max(10, int(combined_data_engineered['Completes'].min()))
                completes_max = min(2000, int(combined_data_engineered['Completes'].max() * 1.2))  # Add some buffer
                # Force min/max to be different
                if completes_min >= completes_max:
                    if completes_max > 110:  # If we can safely subtract
                        completes_min = completes_max - 100
                    else:
                        completes_min = 10
                        completes_max = 500  # Force a sensible range
                completes_default = int((completes_min + completes_max) / 2)
                
                # Final safety check - use hardcoded values if any slider still has equal min/max
                if ir_min >= ir_max:
                    logger.warning(f"IR slider values still equal after fix: min={ir_min}, max={ir_max}")
                    ir_min = 1
                    ir_max = 50
                    ir_default = 25
                
                if loi_min >= loi_max:
                    logger.warning(f"LOI slider values still equal after fix: min={loi_min}, max={loi_max}")
                    loi_min = 5
                    loi_max = 30
                    loi_default = 15
                
                if completes_min >= completes_max:
                    logger.warning(f"Completes slider values still equal after fix: min={completes_min}, max={completes_max}")
                    completes_min = 50
                    completes_max = 500
                    completes_default = 200
                
                # Log final values
                logger.info(f"Final slider values - IR: {ir_min}-{ir_max}, LOI: {loi_min}-{loi_max}, Completes: {completes_min}-{completes_max}")
            
            except Exception as e:
                # Fallback to hardcoded ranges if calculation fails
                logger.error(f"Error calculating slider ranges: {e}")
                ir_min, ir_max, ir_default = 1, 50, 25
                loi_min, loi_max, loi_default = 5, 30, 15
                completes_min, completes_max, completes_default = 50, 500, 200
            
            with col1:
                ir = st.slider(
                    "Incidence Rate (%)", 
                    min_value=ir_min, 
                    max_value=ir_max, 
                    value=ir_default,
                    help="The percentage of people who qualify for your survey"
                )
                
                st.markdown(f"""
                <div style="font-size:0.8em; color:rgba(255, 255, 255, 0.6);">
                    Typical range: {ir_min}% - {ir_max}%<br>
                    <span style="color:{DARK_THEME_COLORS['highlight']}">Lower IR typically increases CPI</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                loi = st.slider(
                    "Length of Interview (min)", 
                    min_value=loi_min, 
                    max_value=loi_max, 
                    value=loi_default,
                    help="How long the survey takes to complete in minutes"
                )
                
                st.markdown(f"""
                <div style="font-size:0.8em; color:rgba(255, 255, 255, 0.6);">
                    Typical range: {loi_min} - {loi_max} minutes<br>
                    <span style="color:{DARK_THEME_COLORS['highlight']}">Longer LOI typically increases CPI</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                completes = st.slider(
                    "Sample Size (Completes)", 
                    min_value=completes_min, 
                    max_value=completes_max, 
                    value=completes_default,
                    help="The number of completed surveys required"
                )
                
                st.markdown(f"""
                <div style="font-size:0.8em; color:rgba(255, 255, 255, 0.6);">
                    Typical range: {completes_min} - {completes_max} completes<br>
                    <span style="color:{DARK_THEME_COLORS['highlight']}">Larger samples may reduce per-unit CPI</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Create user input dictionary
        user_input = {
            'IR': ir,
            'LOI': loi,
            'Completes': completes
        }
        
        # Prediction section with enhanced styling and explanations
        if st.button("Predict CPI", type="primary"):
            with st.spinner("Generating predictions and insights..."):
                # Make predictions
                predictions = predict_cpi(trained_models, user_input, X.columns)
                
                if not predictions:
                    st.error("Failed to generate predictions. Please try different input parameters.")
                    return
                
                # Calculate average prediction
                avg_prediction = sum(predictions.values()) / len(predictions)
                
                # Compare to average CPIs
                won_avg = combined_data_engineered[combined_data_engineered['Type_Won'] == 1]['CPI'].mean()
                lost_avg = combined_data_engineered[combined_data_engineered['Type_Won'] == 0]['CPI'].mean()
                
                # Display predictions with enhanced visualization
                st.subheader("CPI Predictions")
                st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
                # Create enhanced prediction comparison chart
                fig = create_prediction_comparison_with_explanation(predictions, won_avg, lost_avg)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback to standard chart if enhanced version fails
                    fig = create_prediction_comparison_chart(predictions, won_avg, lost_avg)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add explanation of what the chart shows
                with st.expander("📊 Understanding this chart"):
                    st.markdown("""
                    This chart shows:
                    
                    1. **Model Predictions**: Each bar represents a different model's prediction
                    2. **Won/Lost Averages**: The horizontal lines show average CPIs for won (green) and lost (red) bids
                    3. **Ensemble Average**: The dashed white line shows the average of all model predictions
                    
                    **Color Meaning**:
                    - Green/Blue: Predictions below or near the won average (more competitive)
                    - Orange: Predictions between won and lost averages (moderately competitive)
                    - Red: Predictions above the lost average (less competitive)
                    """)
                
                # Display individual predictions in metric cards with custom styling
                st.markdown("<div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>", unsafe_allow_html=True)
                
                # Display model predictions
                for model_name, pred in predictions.items():
                    # Determine styling based on prediction value
                    if pred <= won_avg:
                        color = DARK_THEME_COLORS['won']
                        label = "Competitive"
                    elif pred <= lost_avg:
                        color = DARK_THEME_COLORS['highlight']
                        label = "Moderate"
                    else:
                        color = DARK_THEME_COLORS['lost']
                        label = "High"
                    
                    st.markdown(f"""
                    <div style="flex: 1; min-width: 120px; background-color:rgba(30, 33, 48, 0.7); padding:10px; margin:5px; border-radius:5px; text-align:center;">
                        <p style="font-size:0.9em; margin:0; color:rgba(255, 255, 255, 0.8);">{model_name}</p>
                        <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{color};">${pred:.2f}</p>
                        <p style="font-size:0.8em; margin:0; color:{color};">{label}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display average prediction
                st.markdown(f"""
                <div style="flex: 1; min-width: 150px; background-color:rgba(78, 121, 167, 0.3); padding:10px; margin:5px; border-radius:5px; text-align:center; border:1px solid {DARK_THEME_COLORS['neutral']};">
                    <p style="font-size:0.9em; margin:0; color:rgba(255, 255, 255, 0.9);">Average Prediction</p>
                    <p style="font-size:1.8em; font-weight:bold; margin:5px 0; color:white;">${avg_prediction:.2f}</p>
                    <p style="font-size:0.8em; margin:0; color:rgba(255, 255, 255, 0.7);">{((avg_prediction - won_avg) / won_avg * 100):+.1f}% vs Won Avg</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display comparison and recommendation
                st.subheader("Pricing Recommendation")
                st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
                # Create comparison table with enhanced styling
                comparison_data = {
                    "Metric": ["Won Bids Average", "Lost Bids Average", "Predicted CPI"],
                    "CPI": [f"${won_avg:.2f}", f"${lost_avg:.2f}", f"${avg_prediction:.2f}"],
                    "Difference vs Won Avg": ["0.0%", f"{((lost_avg - won_avg) / won_avg * 100):+.1f}%", f"{((avg_prediction - won_avg) / won_avg * 100):+.1f}%"]
                }
                comparison_df = pd.DataFrame(comparison_data)
                
                # Determine styling for the predicted row
                if avg_prediction <= won_avg:
                    row_color = "rgba(82, 188, 163, 0.2)"  # Green-tinted for competitive
                elif avg_prediction <= lost_avg:
                    row_color = "rgba(242, 142, 43, 0.2)"  # Orange-tinted for moderate
                else:
                    row_color = "rgba(225, 87, 89, 0.2)"   # Red-tinted for high
                
                # Use a different style for the predicted row
                st.dataframe(
                    comparison_df,
                    column_config={
                        "Metric": st.column_config.TextColumn("Metric"),
                        "CPI": st.column_config.TextColumn("CPI"),
                        "Difference vs Won Avg": st.column_config.TextColumn("Difference vs Won Avg")
                    },
                    hide_index=True
                )
                
                # Display recommendation with enhanced styling
                recommendation = get_recommendation(avg_prediction, won_avg, lost_avg)
                
                st.markdown(f"""
                <div style="background-color:rgba(78, 121, 167, 0.2); padding:15px; border-radius:5px; border-left:5px solid #4e79a7; margin-top:20px;">
                    <h3 style="margin-top:0;">Recommendation</h3>
                    <p>{recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Win probability simulation with enhanced visualization
                win_prob = simulate_win_probability(avg_prediction, user_input, won_data, lost_data)
                
                if win_prob:
                    st.subheader("Win Probability Analysis")
                    st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                    
                    # Create columns for layout
                    prob_col1, prob_col2 = st.columns([1, 1])
                    
                    with prob_col1:
                        # Visual probability gauge
                        probability = win_prob['win_probability']
                        
                        # Choose color based on probability
                        if probability >= 70:
                            color = DARK_THEME_COLORS['won']
                        elif probability >= 40:
                            color = DARK_THEME_COLORS['highlight']
                        else:
                            color = DARK_THEME_COLORS['lost']
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Win Probability", 'font': {'size': 24, 'color': 'white'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                                'bar': {'color': color},
                                'bgcolor': "rgba(50, 50, 50, 0.8)",
                                'borderwidth': 2,
                                'bordercolor': "rgba(255, 255, 255, 0.5)",
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(225, 87, 89, 0.3)'},
                                    {'range': [30, 70], 'color': 'rgba(242, 142, 43, 0.3)'},
                                    {'range': [70, 100], 'color': 'rgba(82, 188, 163, 0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': probability
                                }
                            },
                            number={'suffix': "%", 'font': {'size': 28, 'color': 'white'}}
                        ))
                        
                        fig.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=30, r=30, t=30, b=0),
                            height=250
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with prob_col2:
                        # Win probability explanation
                        similar_count = win_prob.get('similar_projects_count', 0)
                        explanation = explain_win_probability(probability, similar_count)
                        st.markdown(explanation)
                    
                    # Price sensitivity curve
                    if 'price_sensitivity' in win_prob and len(win_prob['price_sensitivity']) > 0:
                        price_points = [p[0] for p in win_prob['price_sensitivity']]
                        win_probabilities = [p[1] for p in win_prob['price_sensitivity']]
                        
                        fig = create_price_sensitivity_curve(price_points, win_probabilities, avg_prediction)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Using markdown instead of nested expander
                            st.markdown("""
                            ### 📊 How to use this chart
                            
                            This chart shows how win probability changes at different price points:
                            
                            - **Horizontal Axis**: CPI price points from lowest to highest
                            - **Vertical Axis**: Estimated win probability at each price
                            - **White Dot**: Your current predicted price point
                            
                            **Strategic Use**: 
                            
                            Use this chart to evaluate trade-offs between win probability and profitability.
                            Moving left increases your chances of winning but may reduce profit margins, while
                            moving right increases your potential profit but reduces win probability.
                            """)
                
                # Detailed pricing strategy with enhanced visualization
                st.subheader("Detailed Pricing Strategy")
                st.markdown("<hr style='margin-top: 0; background-color: #4e79a7; height: 2px; margin-bottom: 20px;'>", unsafe_allow_html=True)
                
                # Get detailed strategy
                strategy = get_detailed_pricing_strategy(avg_prediction, user_input, won_data, lost_data)
                
                # Create tabs for different aspects of the strategy
                strategy_tabs = st.tabs(["Pricing Strategy", "Similar Projects", "Factors Explained"])
                
                with strategy_tabs[0]:
                    st.markdown(strategy)
                
                with strategy_tabs[1]:
                    # Similar projects analysis
                    
                    # Filter for similar projects
                    ir_range = 15  # IR range to consider similar
                    loi_range = 5   # LOI range to consider similar
                    
                    similar_won = won_data[
                        (won_data['IR'] >= ir - ir_range) & (won_data['IR'] <= ir + ir_range) &
                        (won_data['LOI'] >= loi - loi_range) & (won_data['LOI'] <= loi + loi_range)
                    ]
                    
                    similar_lost = lost_data[
                        (lost_data['IR'] >= ir - ir_range) & (lost_data['IR'] <= ir + ir_range) &
                        (lost_data['LOI'] >= loi - loi_range) & (lost_data['LOI'] <= loi + loi_range)
                    ]
                    
                    # Create subtabs for similar won and lost projects
                    sim_tabs = st.tabs(["Similar Won Projects", "Similar Lost Projects"])
                    
                    with sim_tabs[0]:
                        if len(similar_won) > 0:
                            st.write(f"Found {len(similar_won)} similar won projects with IR from {ir - ir_range} to {ir + ir_range} and LOI from {loi - loi_range} to {loi + loi_range}.")
                            
                            # Show summary stats with enhanced styling
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div style="background-color:rgba(82, 188, 163, 0.2); padding:10px; border-radius:5px; text-align:center;">
                                    <p style="font-size:0.9em; margin:0;">Average CPI</p>
                                    <p style="font-size:1.8em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['won']};">${similar_won['CPI'].mean():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background-color:rgba(82, 188, 163, 0.1); padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                                    <p style="font-size:0.9em; margin:0;">Median CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['won']};">${similar_won['CPI'].median():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div style="background-color:rgba(82, 188, 163, 0.1); padding:10px; border-radius:5px; text-align:center;">
                                    <p style="font-size:0.9em; margin:0;">Min CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['won']};">${similar_won['CPI'].min():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background-color:rgba(82, 188, 163, 0.1); padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                                    <p style="font-size:0.9em; margin:0;">Max CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['won']};">${similar_won['CPI'].max():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create mini histogram of similar won CPIs
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=similar_won['CPI'],
                                nbinsx=10,
                                marker_color=DARK_THEME_COLORS['won'],
                                opacity=0.7,
                                hovertemplate='CPI: $%{x:.2f}<br>Count: %{y}<extra></extra>'
                            ))
                            
                            # Add line for current prediction
                            fig.add_shape(
                                type="line",
                                x0=avg_prediction,
                                x1=avg_prediction,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(
                                    color="white",
                                    width=2,
                                    dash="dash"
                                )
                            )
                            
                            # Add annotation for prediction line
                            fig.add_annotation(
                                x=avg_prediction,
                                y=1,
                                yref="paper",
                                text="Your Prediction",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="white",
                                arrowwidth=1,
                                arrowsize=0.8,
                                font=dict(color="white"),
                                bgcolor="rgba(0, 0, 0, 0.6)",
                                bordercolor="white",
                                borderpad=4,
                                borderwidth=1
                            )
                            
                            fig.update_layout(
                                title="Distribution of CPIs in Similar Won Projects",
                                xaxis_title="CPI ($)",
                                yaxis_title="Number of Projects",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(
                                    gridcolor='rgba(255,255,255,0.1)',
                                    tickprefix='$'
                                ),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                font=dict(color='white'),
                                margin=dict(l=0, r=0, t=40, b=0),
                                height=250
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show similar projects data table - Using button instead of nested expander
                            if st.button("View Similar Won Projects Data", key="view_won_data"):
                                st.dataframe(similar_won[['IR', 'LOI', 'Completes', 'CPI']], use_container_width=True)
                        else:
                            st.info("No similar won projects found with the specified parameters.")
                    
                    with sim_tabs[1]:
                        if len(similar_lost) > 0:
                            st.write(f"Found {len(similar_lost)} similar lost projects with IR from {ir - ir_range} to {ir + ir_range} and LOI from {loi - loi_range} to {loi + loi_range}.")
                            
                            # Show summary stats with enhanced styling
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div style="background-color:rgba(225, 87, 89, 0.2); padding:10px; border-radius:5px; text-align:center;">
                                    <p style="font-size:0.9em; margin:0;">Average CPI</p>
                                    <p style="font-size:1.8em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['lost']};">${similar_lost['CPI'].mean():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background-color:rgba(225, 87, 89, 0.1); padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                                    <p style="font-size:0.9em; margin:0;">Median CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['lost']};">${similar_lost['CPI'].median():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div style="background-color:rgba(225, 87, 89, 0.1); padding:10px; border-radius:5px; text-align:center;">
                                    <p style="font-size:0.9em; margin:0;">Min CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['lost']};">${similar_lost['CPI'].min():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(f"""
                                <div style="background-color:rgba(225, 87, 89, 0.1); padding:10px; border-radius:5px; text-align:center; margin-top:10px;">
                                    <p style="font-size:0.9em; margin:0;">Max CPI</p>
                                    <p style="font-size:1.6em; font-weight:bold; margin:5px 0; color:{DARK_THEME_COLORS['lost']};">${similar_lost['CPI'].max():.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Create mini histogram of similar lost CPIs
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(
                                x=similar_lost['CPI'],
                                nbinsx=10,
                                marker_color=DARK_THEME_COLORS['lost'],
                                opacity=0.7,
                                hovertemplate='CPI: $%{x:.2f}<br>Count: %{y}<extra></extra>'
                            ))
                            
                            # Add line for current prediction
                            fig.add_shape(
                                type="line",
                                x0=avg_prediction,
                                x1=avg_prediction,
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(
                                    color="white",
                                    width=2,
                                    dash="dash"
                                )
                            )
                            
                            # Add annotation for prediction line
                            fig.add_annotation(
                                x=avg_prediction,
                                y=1,
                                yref="paper",
                                text="Your Prediction",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="white",
                                arrowwidth=1,
                                arrowsize=0.8,
                                font=dict(color="white"),
                                bgcolor="rgba(0, 0, 0, 0.6)",
                                bordercolor="white",
                                borderpad=4,
                                borderwidth=1
                            )
                            
                            fig.update_layout(
                                title="Distribution of CPIs in Similar Lost Projects",
                                xaxis_title="CPI ($)",
                                yaxis_title="Number of Projects",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                xaxis=dict(
                                    gridcolor='rgba(255,255,255,0.1)',
                                    tickprefix='$'
                                ),
                                yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
                                font=dict(color='white'),
                                margin=dict(l=0, r=0, t=40, b=0),
                                height=250
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show similar projects data table - Using button instead of nested expander
                            if st.button("View Similar Lost Projects Data", key="view_lost_data"):
                                st.dataframe(similar_lost[['IR', 'LOI', 'Completes', 'CPI']], use_container_width=True)
                        else:
                            st.info("No similar lost projects found with the specified parameters.")
                
                with strategy_tabs[2]:
                    # Calculate top factors
                    top_factors = []
                    if not feature_importance.empty:
                        top_factors = feature_importance.head(5)[['Feature', 'Importance']].values
                    
                    # Generate explanation of factors
                    factor_explanation = explain_prediction_factors(avg_prediction, user_input, top_factors)
                    st.markdown(factor_explanation)
                    
                    # Add what-if scenarios as regular text - avoiding nested expander
                    st.markdown("""
                    ### 🔍 What-if scenarios
                    
                    Explore how changing each parameter would affect your predicted CPI:
                    
                    - **Lower IR**: Would likely increase your CPI
                    - **Higher IR**: Would likely decrease your CPI
                    - **Shorter LOI**: Would likely decrease your CPI
                    - **Longer LOI**: Would likely increase your CPI
                    - **Smaller Sample**: Would likely increase your per-unit CPI
                    - **Larger Sample**: Would likely decrease your per-unit CPI
                    
                    Try different values in the sliders above and click "Predict CPI" again to see how changes affect the prediction.
                    """)
                
                # Footer with final advice
                st.markdown("""
                <div style="background-color:rgba(38, 39, 48, 0.8); padding:15px; border-radius:5px; margin-top:20px; border:1px solid rgba(78, 121, 167, 0.5);">
                    <h3 style="margin-top:0; color:white;">Final Advice</h3>
                    <p>The predictions provided are based on historical data patterns and should be used as a guide rather than an absolute rule. Always consider your unique market position, client relationship, and strategic goals when finalizing your pricing.</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"Error in prediction component: {e}", exc_info=True)
        st.error(f"An error occurred while rendering the prediction component: {str(e)}")
