"""
Data quality analysis utilities for the CPI Analysis & Prediction Dashboard.
Provides functions for detecting, visualizing, and reporting data quality issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple, Optional
import logging
from utils.theme import COLOR_SYSTEM, DARK_THEME_COLORS, TYPOGRAPHY
from utils.visualization import format_chart_for_dark_mode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_data_quality(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Perform comprehensive data quality analysis on a dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
        dataset_name (str): Name of the dataset (e.g., 'won', 'lost', 'combined')
        
    Returns:
        Dict[str, Any]: Dictionary with data quality metrics and issues
    """
    try:
        # Initialize results dictionary
        results = {
            'dataset_name': dataset_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': {},
            'outliers': {},
            'zeros': {},
            'skewness': {},
            'collinearity': {},
            'data_ranges': {},
            'recommendations': []
        }
        
        # 1. Missing Values Analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        
        # Store missing values info in results
        for col in df.columns:
            count = missing_counts[col]
            percentage = missing_percentages[col]
            results['missing_values'][col] = {
                'count': int(count),
                'percentage': float(percentage),
                'severity': 'high' if percentage > 5 else 'medium' if percentage > 0 else 'low'
            }
            
            # Add recommendation if missing values are significant
            if percentage > 5:
                results['recommendations'].append(
                    f"High missing values ({percentage}%) in column '{col}'. "
                    "Consider imputation or feature engineering."
                )
        
        # 2. Outlier Detection (for numeric columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Using IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Count outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df) * 100).round(2)
            
            # Store outlier info
            results['outliers'][col] = {
                'count': int(outlier_count),
                'percentage': float(outlier_percentage),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'severity': 'high' if outlier_percentage > 10 else 'medium' if outlier_percentage > 5 else 'low'
            }
            
            # Add recommendation if outliers are significant
            if outlier_percentage > 10:
                results['recommendations'].append(
                    f"High outlier percentage ({outlier_percentage}%) in column '{col}'. "
                    "Consider robust scaling or outlier capping."
                )
        
        # 3. Zero Values Detection (for important columns)
        key_cols = ['IR', 'LOI', 'Completes', 'CPI'] if all(col in df.columns for col in ['IR', 'LOI', 'Completes', 'CPI']) else numeric_cols
        for col in key_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                zero_percentage = (zero_count / len(df) * 100).round(2)
                
                # Store zero info
                results['zeros'][col] = {
                    'count': int(zero_count),
                    'percentage': float(zero_percentage),
                    'severity': 'high' if zero_percentage > 5 else 'medium' if zero_percentage > 0 else 'low'
                }
                
                # Add recommendation if zeros are significant in important columns
                if zero_percentage > 5 and col in ['IR', 'LOI', 'Completes', 'CPI']:
                    results['recommendations'].append(
                        f"High zero percentage ({zero_percentage}%) in column '{col}'. "
                        "This may affect modeling performance."
                    )
        
        # 4. Skewness Analysis
        for col in numeric_cols:
            skewness = df[col].skew()
            
            # Store skewness info
            results['skewness'][col] = {
                'value': float(skewness),
                'severity': 'high' if abs(skewness) > 2 else 'medium' if abs(skewness) > 1 else 'low'
            }
            
            # Add recommendation for high skewness
            if abs(skewness) > 2:
                results['recommendations'].append(
                    f"High skewness ({skewness:.2f}) in column '{col}'. "
                    "Consider log or other transformations."
                )
        
        # 5. Collinearity Analysis
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            high_correlations = []
            
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    corr = correlation_matrix.iloc[i, j]
                    
                    if abs(corr) > 0.7:
                        high_correlations.append({
                            'col1': col1,
                            'col2': col2,
                            'correlation': float(corr),
                            'severity': 'high' if abs(corr) > 0.9 else 'medium'
                        })
                        
                        # Add recommendation for high correlation
                        if abs(corr) > 0.9:
                            results['recommendations'].append(
                                f"Very high correlation ({corr:.2f}) between '{col1}' and '{col2}'. "
                                "Consider removing one or creating a composite feature."
                            )
            
            results['collinearity'] = high_correlations
        
        # 6. Data Range Analysis
        for col in numeric_cols:
            results['data_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in data quality analysis: {e}", exc_info=True)
        return {
            'dataset_name': dataset_name,
            'error': str(e)
        }

def calculate_feature_engineering_potential(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate the potential value of various feature engineering techniques.
    
    Args:
        df (pd.DataFrame): Input dataframe to analyze
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary with feature engineering assessments
    """
    try:
        # Initialize results
        results = {}
        
        # Check if key columns exist
        has_key_cols = all(col in df.columns for col in ['IR', 'LOI', 'Completes', 'CPI', 'Type'])
        
        if not has_key_cols:
            return {'error': 'Missing key columns for feature engineering assessment'}
        
        # 1. Assess ratio features
        ratio_features = [
            ('IR_LOI_Ratio', 'IR / LOI', df['IR'] / df['LOI']),
            ('CPI_LOI_Ratio', 'CPI / LOI', df['CPI'] / df['LOI']),
            ('CPI_Completes_Ratio', 'CPI / Completes', df['CPI'] / df['Completes'])
        ]
        
        for name, description, values in ratio_features:
            # Calculate correlation with CPI
            corr_with_cpi = values.corr(df['CPI'])
            
            # Calculate predictive power based on separation between Won/Lost
            if 'Type' in df.columns:
                separation = abs(
                    values[df['Type'] == 'Won'].mean() - 
                    values[df['Type'] == 'Lost'].mean()
                ) / values.std()
            else:
                separation = 0
                
            results[name] = {
                'description': description,
                'correlation_with_cpi': float(corr_with_cpi),
                'won_lost_separation': float(separation),
                'skewness': float(values.skew()),
                'potential': 'high' if (abs(corr_with_cpi) > 0.3 or separation > 0.5) else 'medium' if (abs(corr_with_cpi) > 0.1 or separation > 0.2) else 'low'
            }
        
        # 2. Assess log transforms
        log_features = [
            ('Log_IR', 'Log of IR', np.log1p(df['IR'])),
            ('Log_LOI', 'Log of LOI', np.log1p(df['LOI'])),
            ('Log_Completes', 'Log of Completes', np.log1p(df['Completes'])),
            ('Log_CPI', 'Log of CPI', np.log1p(df['CPI']))
        ]
        
        for name, description, values in log_features:
            # Calculate skewness improvement
            original_skew = df[name.split('_')[1]].skew()
            transformed_skew = values.skew()
            skew_improvement = abs(original_skew) - abs(transformed_skew)
            
            results[name] = {
                'description': description,
                'original_skewness': float(original_skew),
                'transformed_skewness': float(transformed_skew),
                'skewness_improvement': float(skew_improvement),
                'potential': 'high' if skew_improvement > 1 else 'medium' if skew_improvement > 0.5 else 'low'
            }
        
        # 3. Assess interaction terms
        interaction_features = [
            ('IR_LOI_Product', 'IR × LOI', df['IR'] * df['LOI']),
            ('IR_Completes_Product', 'IR × Completes', df['IR'] * df['Completes']),
            ('LOI_Completes_Product', 'LOI × Completes', df['LOI'] * df['Completes'])
        ]
        
        for name, description, values in interaction_features:
            # Calculate correlation with CPI
            corr_with_cpi = values.corr(df['CPI'])
            
            # Calculate predictive power based on separation between Won/Lost
            if 'Type' in df.columns:
                separation = abs(
                    values[df['Type'] == 'Won'].mean() - 
                    values[df['Type'] == 'Lost'].mean()
                ) / values.std()
            else:
                separation = 0
                
            results[name] = {
                'description': description,
                'correlation_with_cpi': float(corr_with_cpi),
                'won_lost_separation': float(separation),
                'potential': 'high' if (abs(corr_with_cpi) > 0.3 or separation > 0.5) else 'medium' if (abs(corr_with_cpi) > 0.1 or separation > 0.2) else 'low'
            }
        
        # 4. Assess efficiency metrics
        efficiency_features = [
            ('CPI_Efficiency', 'CPI / (LOI × IR)', df['CPI'] / (df['LOI'] * df['IR'] / 100)),
            ('CPI_per_Minute', 'CPI / LOI', df['CPI'] / df['LOI'])
        ]
        
        for name, description, values in efficiency_features:
            # Calculate correlation with CPI
            corr_with_cpi = values.corr(df['CPI'])
            
            # Calculate predictive power based on separation between Won/Lost
            if 'Type' in df.columns:
                separation = abs(
                    values[df['Type'] == 'Won'].mean() - 
                    values[df['Type'] == 'Lost'].mean()
                ) / values.std()
            else:
                separation = 0
                
            results[name] = {
                'description': description,
                'correlation_with_cpi': float(corr_with_cpi),
                'won_lost_separation': float(separation),
                'potential': 'high' if (abs(corr_with_cpi) > 0.3 or separation > 0.5) else 'medium' if (abs(corr_with_cpi) > 0.1 or separation > 0.2) else 'low'
            }
        
        return results
    
    except Exception as e:
        logger.error(f"Error in feature engineering assessment: {e}", exc_info=True)
        return {'error': str(e)}

def assess_modeling_suitability(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Assess the suitability of the data for modeling.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        
    Returns:
        Dict[str, Any]: Dictionary with modeling suitability assessments
    """
    try:
        # Initialize results
        results = {
            'data_volume': {},
            'class_balance': {},
            'feature_quality': {},
            'expected_performance': {},
            'recommendations': []
        }
        
        # 1. Data Volume Assessment
        won_count = len(won_data)
        lost_count = len(lost_data)
        total_count = won_count + lost_count
        
        results['data_volume'] = {
            'won_count': won_count,
            'lost_count': lost_count,
            'total_count': total_count,
            'sufficient_for_linear': total_count >= 50,
            'sufficient_for_tree': total_count >= 100,
            'sufficient_for_ensemble': total_count >= 200
        }
        
        # Add recommendations based on data volume
        if total_count < 50:
            results['recommendations'].append(
                "Very limited data volume (< 50 samples). Consider simpler models or collecting more data."
            )
        elif total_count < 100:
            results['recommendations'].append(
                "Limited data volume (< 100 samples). Linear models may perform better than tree-based models."
            )
        elif total_count < 200:
            results['recommendations'].append(
                "Moderate data volume (< 200 samples). Use cross-validation and regularization to prevent overfitting."
            )
        
        # 2. Class Balance Assessment
        won_percentage = (won_count / total_count * 100).round(2)
        lost_percentage = (lost_count / total_count * 100).round(2)
        class_imbalance = abs(won_percentage - lost_percentage)
        
        results['class_balance'] = {
            'won_percentage': float(won_percentage),
            'lost_percentage': float(lost_percentage),
            'imbalance_percentage': float(class_imbalance),
            'is_balanced': class_imbalance < 30
        }
        
        # Add recommendations based on class balance
        if class_imbalance > 50:
            results['recommendations'].append(
                f"Severe class imbalance ({class_imbalance:.1f}%). Consider sampling techniques or specialized metrics."
            )
        elif class_imbalance > 30:
            results['recommendations'].append(
                f"Moderate class imbalance ({class_imbalance:.1f}%). Consider using class weights in models."
            )
        
        # 3. Feature Quality Assessment
        # Combine datasets for correlation analysis
        combined_data = pd.concat([won_data, lost_data])
        
        # Check key modeling features
        key_features = ['IR', 'LOI', 'Completes', 'CPI']
        feature_quality = {}
        
        for feature in key_features:
            if feature in combined_data.columns:
                # Calculate correlation with CPI (for features other than CPI itself)
                if feature != 'CPI':
                    corr_with_cpi = combined_data[feature].corr(combined_data['CPI'])
                else:
                    corr_with_cpi = 1.0
                    
                # Calculate predictive power based on separation between Won/Lost
                if 'Type' in combined_data.columns:
                    won_mean = combined_data[combined_data['Type'] == 'Won'][feature].mean()
                    lost_mean = combined_data[combined_data['Type'] == 'Lost'][feature].mean()
                    feature_std = combined_data[feature].std()
                    separation = abs(won_mean - lost_mean) / feature_std if feature_std > 0 else 0
                else:
                    separation = 0
                
                feature_quality[feature] = {
                    'correlation_with_cpi': float(corr_with_cpi),
                    'won_lost_separation': float(separation),
                    'predictive_power': 'high' if (abs(corr_with_cpi) > 0.3 or separation > 0.5) else 'medium' if (abs(corr_with_cpi) > 0.1 or separation > 0.2) else 'low'
                }
        
        results['feature_quality'] = feature_quality
        
        # Add recommendations based on feature quality
        low_quality_features = [f for f, q in feature_quality.items() if q['predictive_power'] == 'low']
        if low_quality_features:
            results['recommendations'].append(
                f"Low predictive power in features: {', '.join(low_quality_features)}. "
                "Consider feature engineering or collecting additional data."
            )
        
        # 4. Expected Performance Range
        # Estimate based on feature quality and data volume
        feature_scores = [1 if q['predictive_power'] == 'high' else 0.5 if q['predictive_power'] == 'medium' else 0.2 
                         for f, q in feature_quality.items()]
        avg_feature_score = sum(feature_scores) / len(feature_scores) if feature_scores else 0
        
        # Data volume factor (more data = potentially better performance)
        volume_factor = min(1.0, total_count / 500)  # Cap at 500 samples
        
        # Class balance factor (more balanced = potentially better performance)
        balance_factor = 1 - (class_imbalance / 100)
        
        # Combine factors to estimate R² range
        base_r2 = avg_feature_score * 0.6  # Maximum achievable R² with perfect data
        r2_lower = max(0.0, base_r2 * 0.6 * volume_factor * balance_factor)
        r2_upper = min(0.9, base_r2 * 1.2 * volume_factor * balance_factor)
        
        # Estimate RMSE as percentage of mean CPI
        mean_cpi = combined_data['CPI'].mean()
        rmse_lower = mean_cpi * (1 - r2_upper/2)
        rmse_upper = mean_cpi * (1 - r2_lower/2)
        
        results['expected_performance'] = {
            'r2_range': [float(r2_lower), float(r2_upper)],
            'rmse_range': [float(rmse_lower), float(rmse_upper)],
            'confidence': 'high' if (avg_feature_score > 0.7 and volume_factor > 0.7 and balance_factor > 0.7) else 
                         'medium' if (avg_feature_score > 0.4 and volume_factor > 0.4 and balance_factor > 0.4) else 'low'
        }
        
        # Add recommendations based on expected performance
        if results['expected_performance']['confidence'] == 'low':
            results['recommendations'].append(
                "Low confidence in model predictions. Use results cautiously and focus on relative insights rather than absolute values."
            )
        
        return results
    
    except Exception as e:
        logger.error(f"Error in modeling suitability assessment: {e}", exc_info=True)
        return {'error': str(e)}

def identify_key_pricing_factors(combined_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify the key factors influencing CPI pricing in the dataset.
    
    Args:
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
        
    Returns:
        Dict[str, Any]: Dictionary with key pricing factors and insights
    """
    try:
        # Initialize results
        results = {
            'key_factors': [],
            'pricing_thresholds': {},
            'financial_impact': {},
            'recommendations': []
        }
        
        # Feature columns to analyze
        feature_cols = ['IR', 'LOI', 'Completes']
        if not all(col in combined_data.columns for col in feature_cols + ['CPI', 'Type']):
            return {'error': 'Missing required columns for pricing factor analysis'}
        
        # 1. Identify Key Factors by calculating correlation and importance
        factor_insights = {}
        for col in feature_cols:
            # Calculate correlation with CPI
            corr_with_cpi = combined_data[col].corr(combined_data['CPI'])
            
            # Calculate difference between won and lost bids
            won_mean = combined_data[combined_data['Type'] == 'Won'][col].mean()
            lost_mean = combined_data[combined_data['Type'] == 'Lost'][col].mean()
            difference = lost_mean - won_mean
            percent_diff = (difference / won_mean * 100) if won_mean != 0 else 0
            
            # Calculate average CPI by quartiles of this factor
            quartiles = pd.qcut(combined_data[col], 4, labels=False, duplicates='drop')
            combined_data[f'{col}_quartile'] = quartiles
            quartile_cpis = combined_data.groupby(f'{col}_quartile')['CPI'].mean().to_dict()
            
            # Calculate win rate by quartiles
            quartile_win_rates = combined_data.groupby(f'{col}_quartile')['Type'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).to_dict()
            
            # Store insights
            factor_insights[col] = {
                'correlation_with_cpi': float(corr_with_cpi),
                'won_mean': float(won_mean),
                'lost_mean': float(lost_mean),
                'difference': float(difference),
                'percent_difference': float(percent_diff),
                'quartile_cpis': {str(k): float(v) for k, v in quartile_cpis.items()},
                'quartile_win_rates': {str(k): float(v) for k, v in quartile_win_rates.items()},
                'importance_score': float(abs(corr_with_cpi) + abs(percent_diff/100))
            }
        
        # Sort factors by importance score
        sorted_factors = sorted(
            factor_insights.items(), 
            key=lambda x: x[1]['importance_score'], 
            reverse=True
        )
        
        # Store top factors
        results['key_factors'] = [
            {
                'name': factor,
                'importance_score': details['importance_score'],
                'correlation_with_cpi': details['correlation_with_cpi'],
                'won_lost_difference_pct': details['percent_difference'],
                'insights': (
                    f"{'Positive' if details['correlation_with_cpi'] > 0 else 'Negative'} correlation with CPI "
                    f"({details['correlation_with_cpi']:.2f}). "
                    f"{'Higher' if details['difference'] > 0 else 'Lower'} in lost bids by {abs(details['percent_difference']):.1f}%."
                )
            }
            for factor, details in sorted_factors
        ]
        
        # 2. Identify Pricing Thresholds
        for factor, details in factor_insights.items():
            # Find quartile with highest win rate
            best_quartile = max(details['quartile_win_rates'].items(), key=lambda x: x[1])
            
            # Find CPI thresholds by analyzing win rates
            win_rates = []
            for cpi_threshold in np.linspace(combined_data['CPI'].min(), combined_data['CPI'].max(), 20):
                win_rate = (combined_data[combined_data['CPI'] <= cpi_threshold]['Type'] == 'Won').mean() * 100
                win_rates.append((cpi_threshold, win_rate))
            
            # Find threshold with >50% win rate
            threshold_50pct = None
            for threshold, rate in sorted(win_rates, key=lambda x: x[0]):
                if rate >= 50:
                    threshold_50pct = threshold
                    break
            
            results['pricing_thresholds'][factor] = {
                'best_quartile': {
                    'quartile': best_quartile[0],
                    'win_rate': float(best_quartile[1])
                },
                'win_rate_by_cpi': {f"{threshold:.2f}": float(rate) for threshold, rate in win_rates},
                'threshold_50pct_win_rate': float(threshold_50pct) if threshold_50pct is not None else None
            }
        
        # 3. Quantify Financial Impact
        avg_won_cpi = combined_data[combined_data['Type'] == 'Won']['CPI'].mean()
        avg_lost_cpi = combined_data[combined_data['Type'] == 'Lost']['CPI'].mean()
        price_gap = avg_lost_cpi - avg_won_cpi
        price_gap_pct = (price_gap / avg_won_cpi * 100) if avg_won_cpi > 0 else 0
        
        # Calculate potential revenue impact
        avg_won_completes = combined_data[combined_data['Type'] == 'Won']['Completes'].mean()
        avg_lost_completes = combined_data[combined_data['Type'] == 'Lost']['Completes'].mean()
        
        # Simple model: what if we could increase price by 25% of the gap without losing the bid?
        potential_price_increase = price_gap * 0.25
        revenue_impact_per_bid = potential_price_increase * avg_won_completes
        
        # Average bids per month (estimate based on dataset size)
        total_bids_won = (combined_data['Type'] == 'Won').sum()
        avg_bids_per_month = total_bids_won / 12  # Assuming 1 year of data
        
        monthly_revenue_impact = revenue_impact_per_bid * avg_bids_per_month
        
        results['financial_impact'] = {
            'avg_won_cpi': float(avg_won_cpi),
            'avg_lost_cpi': float(avg_lost_cpi),
            'price_gap': float(price_gap),
            'price_gap_percentage': float(price_gap_pct),
            'potential_price_increase': float(potential_price_increase),
            'revenue_impact_per_bid': float(revenue_impact_per_bid),
            'estimated_monthly_impact': float(monthly_revenue_impact)
        }
        
        # Add recommendations
        if price_gap_pct > 20:
            results['recommendations'].append(
                f"Large price gap ({price_gap_pct:.1f}%) between won and lost bids. "
                f"Consider testing prices {potential_price_increase:.2f} higher for similar projects."
            )
        
        return results
    
    except Exception as e:
        logger.error(f"Error in pricing factor analysis: {e}", exc_info=True)
        return {'error': str(e)}

def visualize_data_quality(quality_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create visualizations based on data quality analysis results.
    
    Args:
        quality_results (Dict[str, Any]): Results from analyze_data_quality
        
    Returns:
        Dict[str, go.Figure]: Dictionary of plotly figures
    """
    try:
        figures = {}
        
        # 1. Missing Values Chart
        if 'missing_values' in quality_results:
            missing_data = pd.DataFrame([
                {
                    'Column': col,
                    'Missing Count': details['count'],
                    'Missing Percentage': details['percentage'],
                    'Severity': details['severity'].capitalize()
                }
                for col, details in quality_results['missing_values'].items()
                if details['count'] > 0  # Only show columns with missing values
            ])
            
            if len(missing_data) > 0:
                missing_data = missing_data.sort_values('Missing Percentage', ascending=False)
                
                # Color map based on severity
                color_map = {
                    'Low': COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                    'Medium': COLOR_SYSTEM['SEMANTIC']['WARNING'],
                    'High': COLOR_SYSTEM['SEMANTIC']['ERROR']
                }
                
                fig = go.Figure()
                
                for severity in ['High', 'Medium', 'Low']:
                    filtered_data = missing_data[missing_data['Severity'] == severity]
                    if len(filtered_data) > 0:
                        fig.add_trace(go.Bar(
                            x=filtered_data['Column'],
                            y=filtered_data['Missing Percentage'],
                            name=severity,
                            marker_color=color_map[severity],
                            hovertemplate=(
                                "<b>%{x}</b><br>" +
                                "Missing: %{y:.1f}%<br>" +
                                "Count: %{customdata}<br>" +
                                "Severity: " + severity +
                                "<extra></extra>"
                            ),
                            customdata=filtered_data['Missing Count']
                        ))
                
                fig = format_chart_for_dark_mode(fig, 'Missing Values by Column', height=400)
                
                fig.update_layout(
                    xaxis_title='Column',
                    yaxis_title='Missing Values (%)',
                    yaxis=dict(
                        ticksuffix='%'
                    ),
                    hovermode='closest'
                )
                
                figures['missing_values'] = fig
        
        # 2. Outliers Chart
        if 'outliers' in quality_results:
            outlier_data = pd.DataFrame([
                {
                    'Column': col,
                    'Outlier Count': details['count'],
                    'Outlier Percentage': details['percentage'],
                    'Severity': details['severity'].capitalize()
                }
                for col, details in quality_results['outliers'].items()
                if details['count'] > 0  # Only show columns with outliers
            ])
            
            if len(outlier_data) > 0:
                outlier_data = outlier_data.sort_values('Outlier Percentage', ascending=False)
                
                # Color map based on severity
                color_map = {
                    'Low': COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                    'Medium': COLOR_SYSTEM['SEMANTIC']['WARNING'],
                    'High': COLOR_SYSTEM['SEMANTIC']['ERROR']
                }
                
                fig = go.Figure()
                
                for severity in ['High', 'Medium', 'Low']:
                    filtered_data = outlier_data[outlier_data['Severity'] == severity]
                    if len(filtered_data) > 0:
                        fig.add_trace(go.Bar(
                            x=filtered_data['Column'],
                            y=filtered_data['Outlier Percentage'],
                            name=severity,
                            marker_color=color_map[severity],
                            hovertemplate=(
                                "<b>%{x}</b><br>" +
                                "Outliers: %{y:.1f}%<br>" +
                                "Count: %{customdata}<br>" +
                                "Severity: " + severity +
                                "<extra></extra>"
                            ),
                            customdata=filtered_data['Outlier Count']
                        ))
                
                fig = format_chart_for_dark_mode(fig, 'Outliers by Column', height=400)
                
                fig.update_layout(
                    xaxis_title='Column',
                    yaxis_title='Outliers (%)',
                    yaxis=dict(
                        ticksuffix='%'
                    ),
                    hovermode='closest'
                )
                
                figures['outliers'] = fig
        
        # 3. Skewness Chart
        if 'skewness' in quality_results:
            skewness_data = pd.DataFrame([
                {
                    'Column': col,
                    'Skewness': details['value'],
                    'Severity': details['severity'].capitalize()
                }
                for col, details in quality_results['skewness'].items()
            ])
            
            if len(skewness_data) > 0:
                skewness_data = skewness_data.sort_values('Skewness', key=abs, ascending=False)
                
                # Color map based on severity
                color_map = {
                    'Low': COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                    'Medium': COLOR_SYSTEM['SEMANTIC']['WARNING'],
                    'High': COLOR_SYSTEM['SEMANTIC']['ERROR']
                }
                
                fig = go.Figure()
                
                for severity in ['High', 'Medium', 'Low']:
                    filtered_data = skewness_data[skewness_data['Severity'] == severity]
                    if len(filtered_data) > 0:
                        fig.add_trace(go.Bar(
                            x=filtered_data['Column'],
                            y=filtered_data['Skewness'],
                            name=severity,
                            marker_color=color_map[severity],
                            hovertemplate=(
                                "<b>%{x}</b><br>" +
                                "Skewness: %{y:.2f}<br>" +
                                "Severity: " + severity +
                                "<extra></extra>"
                            )
                        ))
                
                fig = format_chart_for_dark_mode(fig, 'Skewness by Column', height=400)
                
                fig.update_layout(
                    xaxis_title='Column',
                    yaxis_title='Skewness',
                    hovermode='closest'
                )
                
                # Add reference lines for moderate and high skewness
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(skewness_data) - 0.5,
                    y0=1,
                    y1=1,
                    line=dict(
                        color=COLOR_SYSTEM['SEMANTIC']['WARNING'],
                        width=1,
                        dash="dash",
                    )
                )
                
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(skewness_data) - 0.5,
                    y0=-1,
                    y1=-1,
                    line=dict(
                        color=COLOR_SYSTEM['SEMANTIC']['WARNING'],
                        width=1,
                        dash="dash",
                    )
                )
                
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(skewness_data) - 0.5,
                    y0=2,
                    y1=2,
                    line=dict(
                        color=COLOR_SYSTEM['SEMANTIC']['ERROR'],
                        width=1,
                        dash="dash",
                    )
                )
                
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(skewness_data) - 0.5,
                    y0=-2,
                    y1=-2,
                    line=dict(
                        color=COLOR_SYSTEM['SEMANTIC']['ERROR'],
                        width=1,
                        dash="dash",
                    )
                )
                
                figures['skewness'] = fig
        
        # 4. Collinearity Heatmap
        if 'collinearity' in quality_results and quality_results['collinearity']:
            # Extract unique columns from collinearity results
            columns = set()
            for corr in quality_results['collinearity']:
                columns.add(corr['col1'])
                columns.add(corr['col2'])
            
            columns = sorted(list(columns))
            
            # Create correlation matrix
            corr_matrix = pd.DataFrame(index=columns, columns=columns)
            for col in columns:
                corr_matrix.loc[col, col] = 1.0
            
            for corr in quality_results['collinearity']:
                corr_matrix.loc[corr['col1'], corr['col2']] = corr['correlation']
                corr_matrix.loc[corr['col2'], corr['col1']] = corr['correlation']
            
            # Fill NA values with 0
            corr_matrix = corr_matrix.fillna(0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu_r',
                zmin=-1,
                zmax=1,
                hovertemplate=(
                    "<b>%{x} - %{y}</b><br>" +
                    "Correlation: %{z:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
            
            fig = format_chart_for_dark_mode(fig, 'Feature Correlation Heatmap', height=450)
            
            fig.update_layout(
                xaxis_title='',
                yaxis_title='',
                xaxis=dict(
                    tickangle=-45
                ),
                hovermode='closest'
            )
            
            figures['collinearity'] = fig
        
        return figures
    
    except Exception as e:
        logger.error(f"Error in data quality visualization: {e}", exc_info=True)
        return {'error': str(e)}

def visualize_feature_engineering(feature_results: Dict[str, Dict[str, Any]]) -> Dict[str, go.Figure]:
    """
    Create visualizations based on feature engineering potential analysis.
    
    Args:
        feature_results (Dict[str, Dict[str, Any]]): Results from calculate_feature_engineering_potential
        
    Returns:
        Dict[str, go.Figure]: Dictionary of plotly figures
    """
    try:
        figures = {}
        
        # 1. Feature Potential Chart
        if len(feature_results) > 0 and 'error' not in feature_results:
            feature_data = pd.DataFrame([
                {
                    'Feature': name,
                    'Description': details['description'],
                    'Correlation': details.get('correlation_with_cpi', 0),
                    'Separation': details.get('won_lost_separation', 0),
                    'Skew Improvement': details.get('skewness_improvement', 0),
                    'Potential': details['potential'].capitalize()
                }
                for name, details in feature_results.items()
            ])
            
            # Color map based on potential
            color_map = {
                'High': COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                'Medium': COLOR_SYSTEM['SEMANTIC']['WARNING'],
                'Low': COLOR_SYSTEM['SEMANTIC']['ERROR']
            }
            
            # Create a chart showing feature potential by type
            feature_types = [
                ('Ratio Features', [f for f in feature_data['Feature'] if 'Ratio' in f]),
                ('Log Transforms', [f for f in feature_data['Feature'] if 'Log_' in f]),
                ('Interaction Terms', [f for f in feature_data['Feature'] if 'Product' in f]),
                ('Efficiency Metrics', [f for f in feature_data['Feature'] if 'Efficiency' in f or 'per_Minute' in f])
            ]
            
            fig = go.Figure()
            
            for feature_type, features in feature_types:
                type_data = feature_data[feature_data['Feature'].isin(features)]
                
                for potential in ['High', 'Medium', 'Low']:
                    filtered_data = type_data[type_data['Potential'] == potential]
                    if len(filtered_data) > 0:
                        # Determine y-value for the metric based on feature type
                        if 'Log_' in features[0]:
                            y_values = filtered_data['Skew Improvement']
                            y_title = 'Skewness Improvement'
                            hover_template = (
                                "<b>%{customdata[0]}</b><br>" +
                                "Description: %{customdata[1]}<br>" +
                                "Skew Improvement: %{y:.2f}<br>" +
                                "Potential: " + potential +
                                "<extra></extra>"
                            )
                        else:
                            y_values = filtered_data['Correlation'].abs()
                            y_title = 'Absolute Correlation with CPI'
                            hover_template = (
                                "<b>%{customdata[0]}</b><br>" +
                                "Description: %{customdata[1]}<br>" +
                                "Correlation: %{customdata[2]:.2f}<br>" +
                                "Separation: %{customdata[3]:.2f}<br>" +
                                "Potential: " + potential +
                                "<extra></extra>"
                            )
                        
                        fig.add_trace(go.Bar(
                            x=[f"{feature_type}: {f}" for f in filtered_data['Feature']],
                            y=y_values,
                            name=f"{feature_type} - {potential}",
                            marker_color=color_map[potential],
                            hovertemplate=hover_template,
                            customdata=np.column_stack((
                                filtered_data['Feature'],
                                filtered_data['Description'],
                                filtered_data['Correlation'],
                                filtered_data['Separation']
                            ))
                        ))
            
            fig = format_chart_for_dark_mode(fig, 'Feature Engineering Potential', height=500)
            
            fig.update_layout(
                xaxis_title='Feature Type and Name',
                yaxis_title='Metric Value',
                hovermode='closest',
                xaxis=dict(
                    tickangle=-45
                )
            )
            
            figures['feature_potential'] = fig
        
        return figures
    
    except Exception as e:
        logger.error(f"Error in feature engineering visualization: {e}", exc_info=True)
        return {'error': str(e)}

def visualize_modeling_suitability(model_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create visualizations based on modeling suitability assessment.
    
    Args:
        model_results (Dict[str, Any]): Results from assess_modeling_suitability
        
    Returns:
        Dict[str, go.Figure]: Dictionary of plotly figures
    """
    try:
        figures = {}
        
        # 1. Class Balance Chart
        if 'class_balance' in model_results:
            class_balance = model_results['class_balance']
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=['Won Bids', 'Lost Bids'],
                values=[class_balance['won_percentage'], class_balance['lost_percentage']],
                hole=0.4,
                marker=dict(
                    colors=[COLOR_SYSTEM['CHARTS']['WON'], COLOR_SYSTEM['CHARTS']['LOST']],
                    line=dict(color=COLOR_SYSTEM['NEUTRAL']['DARKER'], width=2)
                ),
                textfont=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>" +
                    "Percentage: %{value:.1f}%<br>" +
                    "<extra></extra>"
                )
            ))
            
            # Add annotation for imbalance
            imbalance = class_balance['imbalance_percentage']
            is_balanced = class_balance['is_balanced']
            
            balance_text = (
                f"Class Imbalance: {imbalance:.1f}%<br>" +
                f"<span style='color: {COLOR_SYSTEM['SEMANTIC']['SUCCESS'] if is_balanced else COLOR_SYSTEM['SEMANTIC']['WARNING']}'>Status: {'Balanced' if is_balanced else 'Imbalanced'}</span>"
            )
            
            fig.add_annotation(
                text=balance_text,
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
                )
            )
            
            fig = format_chart_for_dark_mode(fig, 'Class Balance', height=400)
            
            figures['class_balance'] = fig
        
        # 2. Data Volume Adequacy Chart
        if 'data_volume' in model_results:
            data_volume = model_results['data_volume']
            
            # Create data for bar chart
            volume_data = pd.DataFrame([
                {
                    'Metric': 'Total Samples',
                    'Count': data_volume['total_count'],
                    'Sufficient': data_volume['sufficient_for_ensemble'],
                    'Required for Ensemble': 200
                },
                {
                    'Metric': 'Won Samples',
                    'Count': data_volume['won_count'],
                    'Sufficient': data_volume['won_count'] >= 100,
                    'Required for Balance': 100
                },
                {
                    'Metric': 'Lost Samples',
                    'Count': data_volume['lost_count'],
                    'Sufficient': data_volume['lost_count'] >= 100,
                    'Required for Balance': 100
                }
            ])
            
            fig = go.Figure()
            
            # Add bars
            for i, row in volume_data.iterrows():
                color = COLOR_SYSTEM['SEMANTIC']['SUCCESS'] if row['Sufficient'] else COLOR_SYSTEM['SEMANTIC']['WARNING']
                
                fig.add_trace(go.Bar(
                    x=[row['Metric']],
                    y=[row['Count']],
                    name=row['Metric'],
                    marker_color=color,
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Count: %{y}<br>" +
                        f"Required: {row['Required for Ensemble'] if i == 0 else row['Required for Balance']}<br>" +
                        f"Status: {'Sufficient' if row['Sufficient'] else 'Insufficient'}<br>" +
                        "<extra></extra>"
                    )
                ))
            
            # Add threshold lines
            for i, row in volume_data.iterrows():
                threshold = row['Required for Ensemble'] if i == 0 else row['Required for Balance']
                
                fig.add_shape(
                    type="line",
                    x0=i-0.4, x1=i+0.4,
                    y0=threshold, y1=threshold,
                    line=dict(
                        color=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        width=2,
                        dash="dash",
                    )
                )
            
            fig = format_chart_for_dark_mode(fig, 'Data Volume Adequacy', height=400)
            
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Sample Count',
                showlegend=False,
                hovermode='closest'
            )
            
            figures['data_volume'] = fig
        
        # 3. Expected Performance Chart
        if 'expected_performance' in model_results:
            perf = model_results['expected_performance']
            
            fig = go.Figure()
            
            # Add R² range
            fig.add_trace(go.Bar(
                x=['R² Score'],
                y=[(perf['r2_range'][0] + perf['r2_range'][1]) / 2],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[perf['r2_range'][1] - (perf['r2_range'][0] + perf['r2_range'][1]) / 2],
                    arrayminus=[(perf['r2_range'][0] + perf['r2_range'][1]) / 2 - perf['r2_range'][0]]
                ),
                name='R² Score',
                marker_color=COLOR_SYSTEM['PRIMARY']['MAIN'],
                hovertemplate=(
                    "<b>R² Score Range</b><br>" +
                    "Lower bound: %{customdata[0]:.2f}<br>" +
                    "Upper bound: %{customdata[1]:.2f}<br>" +
                    "<extra></extra>"
                ),
                customdata=np.array([[perf['r2_range'][0], perf['r2_range'][1]]])
            ))
            
            # Add RMSE range (scaled to 0-1 for comparison)
            rmse_scaled = [
                (perf['rmse_range'][0] - perf['rmse_range'][0]) / (perf['rmse_range'][1] - perf['rmse_range'][0] + 0.001),
                (perf['rmse_range'][1] - perf['rmse_range'][0]) / (perf['rmse_range'][1] - perf['rmse_range'][0] + 0.001)
            ]
            
            fig.add_trace(go.Bar(
                x=['RMSE (scaled)'],
                y=[(rmse_scaled[0] + rmse_scaled[1]) / 2],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[rmse_scaled[1] - (rmse_scaled[0] + rmse_scaled[1]) / 2],
                    arrayminus=[(rmse_scaled[0] + rmse_scaled[1]) / 2 - rmse_scaled[0]]
                ),
                name='RMSE',
                marker_color=COLOR_SYSTEM['ACCENT']['ORANGE'],
                hovertemplate=(
                    "<b>RMSE Range</b><br>" +
                    "Lower bound: %{customdata[0]:.2f}<br>" +
                    "Upper bound: %{customdata[1]:.2f}<br>" +
                    "<extra></extra>"
                ),
                customdata=np.array([[perf['rmse_range'][0], perf['rmse_range'][1]]])
            ))
            
            # Add confidence indicator
            confidence_color = {
                'high': COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                'medium': COLOR_SYSTEM['SEMANTIC']['WARNING'],
                'low': COLOR_SYSTEM['SEMANTIC']['ERROR']
            }[perf['confidence']]
            
            fig.add_annotation(
                text=f"Model Confidence:<br><b style='color:{confidence_color}'>{perf['confidence'].upper()}</b>",
                x=1.3, y=0.5,
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['PRIMARY']['CONTRAST']
                )
            )
            
            fig = format_chart_for_dark_mode(fig, 'Expected Model Performance', height=400)
            
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Performance Metrics',
                showlegend=False,
                hovermode='closest',
                annotations=[
                    dict(
                        x=0, y=perf['r2_range'][0],
                        xref='x', yref='y',
                        text=f"R² min: {perf['r2_range'][0]:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=COLOR_SYSTEM['PRIMARY']['MAIN'],
                        ax=-60, ay=20,
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        )
                    ),
                    dict(
                        x=0, y=perf['r2_range'][1],
                        xref='x', yref='y',
                        text=f"R² max: {perf['r2_range'][1]:.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=COLOR_SYSTEM['PRIMARY']['MAIN'],
                        ax=-60, ay=-20,
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=12,
                            color=COLOR_SYSTEM['PRIMARY']['MAIN']
                        )
                    )
                ]
            )
            
            figures['expected_performance'] = fig
        
        return figures
    
    except Exception as e:
        logger.error(f"Error in modeling suitability visualization: {e}", exc_info=True)
        return {'error': str(e)}

def visualize_pricing_factors(factor_results: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create visualizations based on key pricing factors analysis.
    
    Args:
        factor_results (Dict[str, Any]): Results from identify_key_pricing_factors
        
    Returns:
        Dict[str, go.Figure]: Dictionary of plotly figures
    """
    try:
        figures = {}
        
        # 1. Key Factors Importance Chart
        if 'key_factors' in factor_results:
            factors = factor_results['key_factors']
            
            if factors:
                # Create data for bar chart
                factor_data = pd.DataFrame([
                    {
                        'Factor': f['name'],
                        'Importance Score': f['importance_score'],
                        'Correlation': f['correlation_with_cpi'],
                        'Difference %': f['won_lost_difference_pct']
                    }
                    for f in factors
                ]).sort_values('Importance Score', ascending=False)
                
                fig = go.Figure()
                
                # Add bars for importance score
                fig.add_trace(go.Bar(
                    x=factor_data['Factor'],
                    y=factor_data['Importance Score'],
                    name='Importance Score',
                    marker_color=COLOR_SYSTEM['PRIMARY']['MAIN'],
                    hovertemplate=(
                        "<b>%{x}</b><br>" +
                        "Importance Score: %{y:.2f}<br>" +
                        "Correlation with CPI: %{customdata[0]:.2f}<br>" +
                        "Won/Lost Difference: %{customdata[1]:.1f}%<br>" +
                        "<extra></extra>"
                    ),
                    customdata=np.column_stack((
                        factor_data['Correlation'],
                        factor_data['Difference %']
                    ))
                ))
                
                fig = format_chart_for_dark_mode(fig, 'Key Pricing Factors', height=400)
                
                fig.update_layout(
                    xaxis_title='Factor',
                    yaxis_title='Importance Score',
                    showlegend=False,
                    hovermode='closest'
                )
                
                figures['key_factors'] = fig
        
        # 2. Financial Impact Chart
        if 'financial_impact' in factor_results:
            impact = factor_results['financial_impact']
            
            fig = go.Figure()
            
            # Add bar for CPI comparison
            fig.add_trace(go.Bar(
                x=['Won Bids', 'Lost Bids'],
                y=[impact['avg_won_cpi'], impact['avg_lost_cpi']],
                marker_color=[COLOR_SYSTEM['CHARTS']['WON'], COLOR_SYSTEM['CHARTS']['LOST']],
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Average CPI: $%{y:.2f}<br>" +
                    "<extra></extra>"
                )
            ))
            
            # Add annotation for price gap
            fig.add_annotation(
                x=0.5, y=impact['avg_lost_cpi'] * 1.1,
                text=(
                    f"Price Gap: ${impact['price_gap']:.2f}<br>" +
                    f"({impact['price_gap_percentage']:.1f}%)"
                ),
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['NEUTRAL']['LIGHT']
                )
            )
            
            # Add annotation for potential revenue impact
            fig.add_annotation(
                x=1.5, y=impact['avg_won_cpi'],
                xanchor='left',
                text=(
                    f"Potential Price Increase: ${impact['potential_price_increase']:.2f}<br>" +
                    f"Est. Revenue Impact/Bid: ${impact['revenue_impact_per_bid']:.2f}<br>" +
                    f"Est. Monthly Impact: ${impact['estimated_monthly_impact']:.2f}"
                ),
                showarrow=False,
                font=dict(
                    family=TYPOGRAPHY['FONT_FAMILY'],
                    size=14,
                    color=COLOR_SYSTEM['SEMANTIC']['SUCCESS']
                ),
                bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKEST'],
                borderpad=4,
                bordercolor=COLOR_SYSTEM['NEUTRAL']['DARKER'],
                borderwidth=1
            )
            
            fig = format_chart_for_dark_mode(fig, 'CPI Pricing Gap Analysis', height=450)
            
            fig.update_layout(
                xaxis_title='',
                yaxis_title='Average CPI ($)',
                yaxis=dict(
                    tickprefix='$'
                ),
                showlegend=False,
                hovermode='closest'
            )
            
            figures['financial_impact'] = fig
        
        # 3. Win Rate by CPI Threshold Chart
        if 'pricing_thresholds' in factor_results:
            thresholds = factor_results['pricing_thresholds']
            
            if thresholds:
                # Get the threshold data for the first factor (usually the most important one)
                first_factor = list(thresholds.keys())[0]
                threshold_data = thresholds[first_factor]
                
                # Extract win rate by CPI data
                win_rates = threshold_data['win_rate_by_cpi']
                
                win_rate_data = pd.DataFrame([
                    {
                        'CPI Threshold': float(cpi),
                        'Win Rate (%)': rate
                    }
                    for cpi, rate in win_rates.items()
                ]).sort_values('CPI Threshold')
                
                fig = go.Figure()
                
                # Add line for win rate
                fig.add_trace(go.Scatter(
                    x=win_rate_data['CPI Threshold'],
                    y=win_rate_data['Win Rate (%)'],
                    mode='lines+markers',
                    line=dict(
                        color=COLOR_SYSTEM['PRIMARY']['MAIN'],
                        width=3
                    ),
                    marker=dict(
                        size=8,
                        color=COLOR_SYSTEM['PRIMARY']['MAIN'],
                        line=dict(
                            width=2,
                            color=COLOR_SYSTEM['NEUTRAL']['DARKER']
                        )
                    ),
                    hovertemplate=(
                        "<b>CPI Threshold: $%{x:.2f}</b><br>" +
                        "Win Rate: %{y:.1f}%<br>" +
                        "<extra></extra>"
                    )
                ))
                
                # Add 50% win rate threshold if available
                if threshold_data['threshold_50pct_win_rate']:
                    threshold_50 = threshold_data['threshold_50pct_win_rate']
                    
                    fig.add_shape(
                        type="line",
                        x0=threshold_50,
                        x1=threshold_50,
                        y0=0,
                        y1=50,
                        line=dict(
                            color=COLOR_SYSTEM['SEMANTIC']['SUCCESS'],
                            width=2,
                            dash="dash",
                        )
                    )
                    
                    fig.add_annotation(
                        x=threshold_50,
                        y=55,
                        text=f"50% Win Rate<br>Threshold: ${threshold_50:.2f}",
                        showarrow=False,
                        font=dict(
                            family=TYPOGRAPHY['FONT_FAMILY'],
                            size=14,
                            color=COLOR_SYSTEM['SEMANTIC']['SUCCESS']
                        ),
                        bgcolor=COLOR_SYSTEM['NEUTRAL']['DARKEST'],
                        borderpad=4
                    )
                
                # Add horizontal line at 50% win rate
                fig.add_shape(
                    type="line",
                    x0=win_rate_data['CPI Threshold'].min(),
                    x1=win_rate_data['CPI Threshold'].max(),
                    y0=50,
                    y1=50,
                    line=dict(
                        color=COLOR_SYSTEM['NEUTRAL']['LIGHT'],
                        width=1,
                        dash="dash",
                    )
                )
                
                fig = format_chart_for_dark_mode(fig, f'Win Rate by CPI Threshold ({first_factor})', height=450)
                
                fig.update_layout(
                    xaxis_title='CPI Threshold ($)',
                    yaxis_title='Win Rate (%)',
                    xaxis=dict(
                        tickprefix='$'
                    ),
                    yaxis=dict(
                        ticksuffix='%'
                    ),
                    showlegend=False,
                    hovermode='closest'
                )
                
                figures['win_rate_by_cpi'] = fig
        
        return figures
    
    except Exception as e:
        logger.error(f"Error in pricing factors visualization: {e}", exc_info=True)
        return {'error': str(e)}

def show_data_quality_analysis(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Show data quality analysis in Streamlit.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        dataset_name (str): Name of the dataset
    """
    try:
        st.markdown(f"## Data Quality Analysis: {dataset_name}")
        
        # Perform data quality analysis
        quality_results = analyze_data_quality(df, dataset_name)
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Missing & Outliers", "Distributions", "Correlations"])
        
        with tab1:
            # Overview of data quality
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Rows", quality_results['row_count'])
                
            with col2:
                st.metric("Total Columns", quality_results['column_count'])
                
            with col3:
                missing_rate = sum(d['percentage'] for d in quality_results['missing_values'].values()) / len(quality_results['missing_values'])
                severity = "🟢 Low" if missing_rate < 1 else "🟡 Medium" if missing_rate < 5 else "🔴 High"
                st.metric("Missing Data Rate", f"{missing_rate:.2f}%", help=f"Overall missing data rate across all columns. {severity}")
            
            # Key recommendations
            if quality_results['recommendations']:
                st.markdown("### Key Recommendations")
                for i, rec in enumerate(quality_results['recommendations']):
                    st.markdown(f"{i+1}. {rec}")
            else:
                st.success("No major data quality issues detected.")
        
        with tab2:
            # Visualize missing values and outliers
            figures = visualize_data_quality(quality_results)
            
            if 'missing_values' in figures:
                st.markdown("### Missing Values")
                st.plotly_chart(figures['missing_values'], use_container_width=True)
            
            if 'outliers' in figures:
                st.markdown("### Outliers")
                st.plotly_chart(figures['outliers'], use_container_width=True)
            
            # Display tables with details
            with st.expander("Detailed Missing Values"):
                missing_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Missing Count': details['count'],
                        'Missing Percentage': f"{details['percentage']:.2f}%",
                        'Severity': details['severity'].capitalize()
                    }
                    for col, details in quality_results['missing_values'].items()
                ]).sort_values('Missing Count', ascending=False)
                
                st.dataframe(missing_df)
            
            with st.expander("Detailed Outlier Analysis"):
                outlier_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Outlier Count': details['count'],
                        'Outlier Percentage': f"{details['percentage']:.2f}%",
                        'Lower Bound': f"{details['lower_bound']:.2f}",
                        'Upper Bound': f"{details['upper_bound']:.2f}",
                        'Severity': details['severity'].capitalize()
                    }
                    for col, details in quality_results['outliers'].items()
                ]).sort_values('Outlier Count', ascending=False)
                
                st.dataframe(outlier_df)
                
            with st.expander("Zeros in Key Columns"):
                zeros_df = pd.DataFrame([
                    {
                        'Column': col,
                        'Zero Count': details['count'],
                        'Zero Percentage': f"{details['percentage']:.2f}%",
                        'Severity': details['severity'].capitalize()
                    }
                    for col, details in quality_results['zeros'].items()
                ]).sort_values('Zero Count', ascending=False)
                
                st.dataframe(zeros_df)
        
        with tab3:
            # Visualize distributions and skewness
            if 'skewness' in figures:
                st.markdown("### Skewness by Column")
                st.plotly_chart(figures['skewness'], use_container_width=True)
                
                st.markdown("""
                **Skewness Interpretation:**
                - **< -2 or > 2**: High skewness (consider log transform)
                - **< -1 or > 1**: Moderate skewness
                - **-1 to 1**: Approximately symmetrical
                """)
            
            # Data ranges table
            st.markdown("### Data Ranges")
            ranges_df = pd.DataFrame([
                {
                    'Column': col,
                    'Min': f"{details['min']:.2f}",
                    'Max': f"{details['max']:.2f}",
                    'Mean': f"{details['mean']:.2f}",
                    'Median': f"{details['median']:.2f}",
                    'Std Dev': f"{details['std']:.2f}"
                }
                for col, details in quality_results['data_ranges'].items()
            ])
            
            st.dataframe(ranges_df)
        
        with tab4:
            # Visualize correlations
            if 'collinearity' in figures:
                st.markdown("### Feature Correlations")
                st.plotly_chart(figures['collinearity'], use_container_width=True)
            
            # Correlation details
            if quality_results['collinearity']:
                st.markdown("### High Correlations")
                corr_df = pd.DataFrame([
                    {
                        'Feature 1': corr['col1'],
                        'Feature 2': corr['col2'],
                        'Correlation': f"{corr['correlation']:.2f}",
                        'Severity': corr['severity'].capitalize()
                    }
                    for corr in quality_results['collinearity']
                ]).sort_values('Correlation', key=abs, ascending=False)
                
                st.dataframe(corr_df)
                
                st.markdown("""
                **Correlation Interpretation:**
                - **> 0.9**: Very high correlation (consider removing one feature)
                - **0.7-0.9**: High correlation (watch for multicollinearity)
                - **0.5-0.7**: Moderate correlation
                - **< 0.5**: Low correlation
                """)
            else:
                st.info("No high correlations detected between features.")
    
    except Exception as e:
        logger.error(f"Error showing data quality analysis: {e}", exc_info=True)
        st.error(f"Error analyzing data quality: {str(e)}")

def show_feature_engineering_potential(df: pd.DataFrame) -> None:
    """
    Show feature engineering potential analysis in Streamlit.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    try:
        st.markdown("## Feature Engineering Potential")
        
        # Perform feature engineering potential analysis
        feature_results = calculate_feature_engineering_potential(df)
        
        if 'error' in feature_results:
            st.error(feature_results['error'])
            return
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "Implementation Guide"])
        
        with tab1:
            # Visualize feature potential
            figures = visualize_feature_engineering(feature_results)
            
            if 'feature_potential' in figures:
                st.plotly_chart(figures['feature_potential'], use_container_width=True)
            
            # Key insights
            high_potential = [name for name, details in feature_results.items() if details['potential'] == 'high']
            medium_potential = [name for name, details in feature_results.items() if details['potential'] == 'medium']
            
            if high_potential:
                st.success(f"### High Potential Features\n{', '.join(high_potential)}")
            
            if medium_potential:
                st.warning(f"### Medium Potential Features\n{', '.join(medium_potential)}")
            
            # Recommendations
            st.markdown("### Feature Engineering Recommendations")
            recs = []
            
            # Recommend log transforms for high skewness
            log_features = [name for name, details in feature_results.items() 
                          if name.startswith('Log_') and details['potential'] == 'high']
            if log_features:
                original_cols = [name.split('_')[1] for name in log_features]
                recs.append(f"Apply log transformations to columns with high skewness: {', '.join(original_cols)}")
            
            # Recommend ratio features with good separation
            ratio_features = [name for name, details in feature_results.items() 
                            if 'Ratio' in name and details['potential'] in ['high', 'medium']]
            if ratio_features:
                recs.append(f"Create ratio features: {', '.join(ratio_features)}")
            
            # Recommend interaction terms with good correlation
            interaction_features = [name for name, details in feature_results.items() 
                                  if 'Product' in name and details['potential'] in ['high', 'medium']]
            if interaction_features:
                recs.append(f"Create interaction terms: {', '.join(interaction_features)}")
            
            # Recommend efficiency metrics
            efficiency_features = [name for name, details in feature_results.items() 
                                 if ('Efficiency' in name or 'per_Minute' in name) and details['potential'] in ['high', 'medium']]
            if efficiency_features:
                recs.append(f"Create efficiency metrics: {', '.join(efficiency_features)}")
            
            for i, rec in enumerate(recs):
                st.markdown(f"{i+1}. {rec}")
        
        with tab2:
            # Detailed tables by feature type
            st.markdown("### Ratio Features")
            ratio_df = pd.DataFrame([
                {
                    'Feature': name,
                    'Description': details['description'],
                    'Correlation with CPI': f"{details.get('correlation_with_cpi', 0):.2f}",
                    'Won/Lost Separation': f"{details.get('won_lost_separation', 0):.2f}",
                    'Potential': details['potential'].capitalize()
                }
                for name, details in feature_results.items()
                if 'Ratio' in name
            ]).sort_values('Potential', key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2}))
            
            st.dataframe(ratio_df)
            
            st.markdown("### Log Transformations")
            log_df = pd.DataFrame([
                {
                    'Feature': name,
                    'Description': details['description'],
                    'Original Skewness': f"{details['original_skewness']:.2f}",
                    'Transformed Skewness': f"{details['transformed_skewness']:.2f}",
                    'Improvement': f"{details['skewness_improvement']:.2f}",
                    'Potential': details['potential'].capitalize()
                }
                for name, details in feature_results.items()
                if name.startswith('Log_')
            ]).sort_values('Improvement', ascending=False)
            
            st.dataframe(log_df)
            
            st.markdown("### Interaction Features")
            interaction_df = pd.DataFrame([
                {
                    'Feature': name,
                    'Description': details['description'],
                    'Correlation with CPI': f"{details.get('correlation_with_cpi', 0):.2f}",
                    'Won/Lost Separation': f"{details.get('won_lost_separation', 0):.2f}",
                    'Potential': details['potential'].capitalize()
                }
                for name, details in feature_results.items()
                if 'Product' in name
            ]).sort_values('Potential', key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2}))
            
            st.dataframe(interaction_df)
            
            st.markdown("### Efficiency Metrics")
            efficiency_df = pd.DataFrame([
                {
                    'Feature': name,
                    'Description': details['description'],
                    'Correlation with CPI': f"{details.get('correlation_with_cpi', 0):.2f}",
                    'Won/Lost Separation': f"{details.get('won_lost_separation', 0):.2f}",
                    'Potential': details['potential'].capitalize()
                }
                for name, details in feature_results.items()
                if 'Efficiency' in name or 'per_Minute' in name
            ]).sort_values('Potential', key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2}))
            
            st.dataframe(efficiency_df)
        
        with tab3:
            # Implementation guide
            st.markdown("### Feature Implementation Code")
            
            # Select high and medium potential features
            good_features = {name: details for name, details in feature_results.items() 
                           if details['potential'] in ['high', 'medium']}
            
            if good_features:
                st.markdown("Add the following feature engineering code to your preprocessing pipeline:")
                
                code_blocks = []
                
                # Group by feature type
                ratio_code = "\n".join([f"df['{name}'] = {details['description']}" 
                                     for name, details in good_features.items() if 'Ratio' in name])
                if ratio_code:
                    code_blocks.append("# Ratio Features\n" + ratio_code)
                
                log_code = "\n".join([f"df['{name}'] = np.log1p(df['{name.split('_')[1]}'])" 
                                   for name, details in good_features.items() if name.startswith('Log_')])
                if log_code:
                    code_blocks.append("# Log Transformations\n" + log_code)
                
                interaction_code = "\n".join([f"df['{name}'] = {details['description']}" 
                                          for name, details in good_features.items() if 'Product' in name])
                if interaction_code:
                    code_blocks.append("# Interaction Features\n" + interaction_code)
                
                efficiency_code = "\n".join([f"df['{name}'] = {details['description']}" 
                                         for name, details in good_features.items() 
                                         if 'Efficiency' in name or 'per_Minute' in name])
                if efficiency_code:
                    code_blocks.append("# Efficiency Metrics\n" + efficiency_code)
                
                full_code = "\n\n".join(code_blocks)
                
                st.code(f"""def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add engineered features based on data analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe with original features
        
    Returns:
        pd.DataFrame: DataFrame with added engineered features
    '''
    import numpy as np
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    {full_code}
    
    return df
""", language="python")
            else:
                st.info("No high or medium potential features identified for implementation.")
    
    except Exception as e:
        logger.error(f"Error showing feature engineering potential: {e}", exc_info=True)
        st.error(f"Error analyzing feature engineering potential: {str(e)}")

def show_modeling_suitability(won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Show modeling suitability analysis in Streamlit.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    """
    try:
        st.markdown("## Modeling Suitability Analysis")
        
        # Perform modeling suitability analysis
        model_results = assess_modeling_suitability(won_data, lost_data)
        
        if 'error' in model_results:
            st.error(model_results['error'])
            return
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Overview", "Data Adequacy", "Performance Expectations"])
        
        with tab1:
            # Key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                volume = model_results['data_volume']
                status = "🟢 Sufficient" if volume['sufficient_for_ensemble'] else "🟡 Limited" if volume['sufficient_for_linear'] else "🔴 Insufficient"
                st.metric("Data Volume", f"{volume['total_count']} samples", help=f"Status: {status}")
            
            with col2:
                balance = model_results['class_balance']
                status = "🟢 Balanced" if balance['is_balanced'] else "🟡 Imbalanced"
                st.metric("Class Balance", f"{balance['imbalance_percentage']:.1f}% diff", help=f"Status: {status}")
            
            with col3:
                perf = model_results['expected_performance']
                st.metric("Expected R²", f"{perf['r2_range'][0]:.2f}-{perf['r2_range'][1]:.2f}", help=f"Confidence: {perf['confidence'].capitalize()}")
            
            # Key recommendations
            if model_results['recommendations']:
                st.markdown("### Key Recommendations")
                for i, rec in enumerate(model_results['recommendations']):
                    st.markdown(f"{i+1}. {rec}")
        
        with tab2:
            # Data volume and class balance visualizations
            figures = visualize_modeling_suitability(model_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'data_volume' in figures:
                    st.markdown("### Data Volume Assessment")
                    st.plotly_chart(figures['data_volume'], use_container_width=True)
            
            with col2:
                if 'class_balance' in figures:
                    st.markdown("### Class Balance")
                    st.plotly_chart(figures['class_balance'], use_container_width=True)
            
            # Feature quality table
            st.markdown("### Feature Quality Assessment")
            feature_quality = model_results['feature_quality']
            
            feature_df = pd.DataFrame([
                {
                    'Feature': feature,
                    'Correlation with CPI': f"{details['correlation_with_cpi']:.2f}",
                    'Won/Lost Separation': f"{details['won_lost_separation']:.2f}",
                    'Predictive Power': details['predictive_power'].capitalize()
                }
                for feature, details in feature_quality.items()
            ]).sort_values('Predictive Power', key=lambda x: x.map({'High': 0, 'Medium': 1, 'Low': 2}))
            
            st.dataframe(feature_df)
            
            # Model recommendations based on data
            st.markdown("### Recommended Models")
            
            vol = model_results['data_volume']
            bal = model_results['class_balance']
            
            models = []
            
            if vol['sufficient_for_ensemble']:
                models.append({
                    'Model': 'Gradient Boosting',
                    'Suitability': 'High',
                    'Notes': 'Robust to outliers and can capture complex relationships'
                })
                models.append({
                    'Model': 'Random Forest',
                    'Suitability': 'High',
                    'Notes': 'Good for feature importance and resistant to overfitting'
                })
            
            if vol['sufficient_for_linear']:
                models.append({
                    'Model': 'Ridge Regression',
                    'Suitability': 'Medium' if vol['sufficient_for_ensemble'] else 'High',
                    'Notes': 'Good for interpretability and handling multicollinearity'
                })
            
            if not bal['is_balanced']:
                models.append({
                    'Model': 'Weighted Models',
                    'Suitability': 'High',
                    'Notes': 'Use class weights to address imbalanced data'
                })
            
            models_df = pd.DataFrame(models)
            st.dataframe(models_df)
        
        with tab3:
            # Performance expectations visualization
            if 'expected_performance' in figures:
                st.markdown("### Expected Model Performance")
                st.plotly_chart(figures['expected_performance'], use_container_width=True)
            
            # Performance details
            perf = model_results['expected_performance']
            
            st.markdown("### Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### R² Score Range")
                st.markdown(f"**Lower bound:** {perf['r2_range'][0]:.2f}")
                st.markdown(f"**Upper bound:** {perf['r2_range'][1]:.2f}")
                st.markdown(f"**Confidence:** {perf['confidence'].capitalize()}")
                
                # Interpretation
                r2_interpret = ""
                if perf['r2_range'][1] > 0.7:
                    r2_interpret = "Strong predictive power"
                elif perf['r2_range'][1] > 0.5:
                    r2_interpret = "Moderate predictive power"
                else:
                    r2_interpret = "Limited predictive power"
                
                st.markdown(f"**Interpretation:** {r2_interpret}")
            
            with col2:
                st.markdown("#### RMSE Range")
                st.markdown(f"**Lower bound:** ${perf['rmse_range'][0]:.2f}")
                st.markdown(f"**Upper bound:** ${perf['rmse_range'][1]:.2f}")
                
                # Context
                combined_df = pd.concat([won_data, lost_data])
                avg_cpi = combined_df['CPI'].mean()
                rmse_pct = ((perf['rmse_range'][0] + perf['rmse_range'][1]) / 2) / avg_cpi * 100
                
                st.markdown(f"**Average CPI:** ${avg_cpi:.2f}")
                st.markdown(f"**RMSE as % of Average CPI:** {rmse_pct:.1f}%")
                
                # Interpretation
                rmse_interpret = ""
                if rmse_pct < 20:
                    rmse_interpret = "High prediction accuracy"
                elif rmse_pct < 30:
                    rmse_interpret = "Moderate prediction accuracy"
                else:
                    rmse_interpret = "Limited prediction accuracy"
                
                st.markdown(f"**Interpretation:** {rmse_interpret}")
    
    except Exception as e:
        logger.error(f"Error showing modeling suitability: {e}", exc_info=True)
        st.error(f"Error analyzing modeling suitability: {str(e)}")

def show_pricing_factors(combined_data: pd.DataFrame) -> None:
    """
    Show key pricing factors analysis in Streamlit.
    
    Args:
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    try:
        st.markdown("## Key Pricing Factors Analysis")
        
        # Perform key pricing factors analysis
        factor_results = identify_key_pricing_factors(combined_data)
        
        if 'error' in factor_results:
            st.error(factor_results['error'])
            return
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Key Drivers", "Pricing Thresholds", "Financial Impact"])
        
        with tab1:
            # Key factors visualization
            figures = visualize_pricing_factors(factor_results)
            
            if 'key_factors' in figures:
                st.plotly_chart(figures['key_factors'], use_container_width=True)
            
            # Key factors table
            st.markdown("### Detailed Factor Analysis")
            
            factors_df = pd.DataFrame([
                {
                    'Factor': f['name'],
                    'Importance Score': f"{f['importance_score']:.2f}",
                    'Correlation with CPI': f"{f['correlation_with_cpi']:.2f}",
                    'Won/Lost Difference': f"{f['won_lost_difference_pct']:.1f}%",
                    'Insights': f['insights']
                }
                for f in factor_results['key_factors']
            ])
            
            st.dataframe(factors_df)
            
            # Key insights
            if factor_results['key_factors']:
                st.markdown("### Key Insights")
                
                for i, factor in enumerate(factor_results['key_factors'][:3]):  # Top 3 factors
                    st.markdown(f"**{i+1}. {factor['name']}:** {factor['insights']}")
        
        with tab2:
            # Win rate by CPI visualization
            if 'win_rate_by_cpi' in figures:
                st.plotly_chart(figures['win_rate_by_cpi'], use_container_width=True)
            
            # Pricing thresholds table
            st.markdown("### Pricing Thresholds by Factor")
            
            thresholds = factor_results['pricing_thresholds']
            thresholds_rows = []
            
            for factor, details in thresholds.items():
                # Get best quartile info
                best_quartile = details['best_quartile']
                
                # Get 50% win rate threshold
                threshold_50 = details['threshold_50pct_win_rate']
                
                thresholds_rows.append({
                    'Factor': factor,
                    'Best Quartile': best_quartile['quartile'],
                    'Best Quartile Win Rate': f"{best_quartile['win_rate']:.1f}%",
                    '50% Win Rate Threshold': f"${threshold_50:.2f}" if threshold_50 is not None else "N/A"
                })
            
            thresholds_df = pd.DataFrame(thresholds_rows)
            st.dataframe(thresholds_df)
            
            # Pricing recommendations
            st.markdown("### Pricing Recommendations")
            
            # Get top factor
            top_factor = factor_results['key_factors'][0]['name'] if factor_results['key_factors'] else None
            
            if top_factor and top_factor in thresholds:
                top_details = thresholds[top_factor]
                threshold_50 = top_details['threshold_50pct_win_rate']
                
                if threshold_50 is not None:
                    st.success(f"For optimal pricing with balanced win probability, aim for CPI around **${threshold_50:.2f}** when {top_factor} is a key factor.")
                
                # Get win rates at different CPI points
                win_rates = top_details['win_rate_by_cpi']
                win_rate_points = sorted([(float(cpi), rate) for cpi, rate in win_rates.items()])
                
                st.markdown("Win probability at different price points:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    low_idx = len(win_rate_points) // 4
                    low_cpi, low_rate = win_rate_points[low_idx]
                    st.metric("Low Price Point", f"${low_cpi:.2f}", f"{low_rate:.1f}% win rate")
                
                with col2:
                    mid_idx = len(win_rate_points) // 2
                    mid_cpi, mid_rate = win_rate_points[mid_idx]
                    st.metric("Medium Price Point", f"${mid_cpi:.2f}", f"{mid_rate:.1f}% win rate")
                
                with col3:
                    high_idx = 3 * len(win_rate_points) // 4
                    high_cpi, high_rate = win_rate_points[high_idx]
                    st.metric("High Price Point", f"${high_cpi:.2f}", f"{high_rate:.1f}% win rate")
        
        with tab3:
            # Financial impact visualization
            if 'financial_impact' in figures:
                st.plotly_chart(figures['financial_impact'], use_container_width=True)
            
            # Financial impact details
            impact = factor_results['financial_impact']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### CPI Gap Analysis")
                st.markdown(f"**Won bids average CPI:** ${impact['avg_won_cpi']:.2f}")
                st.markdown(f"**Lost bids average CPI:** ${impact['avg_lost_cpi']:.2f}")
                st.markdown(f"**Price gap:** ${impact['price_gap']:.2f} ({impact['price_gap_percentage']:.1f}%)")
            
            with col2:
                st.markdown("### Revenue Impact")
                st.markdown(f"**Potential price increase:** ${impact['potential_price_increase']:.2f} per CPI")
                st.markdown(f"**Revenue impact per bid:** ${impact['revenue_impact_per_bid']:.2f}")
                st.markdown(f"**Estimated monthly impact:** ${impact['estimated_monthly_impact']:.2f}")
            
            # Strategic recommendations
            st.markdown("### Strategic Pricing Recommendations")
            
            price_gap_pct = impact['price_gap_percentage']
            monthly_impact = impact['estimated_monthly_impact']
            
            if price_gap_pct > 20:
                st.success(f"""
                ### Significant pricing opportunity identified
                
                Your won bids are priced **{price_gap_pct:.1f}%** lower than lost bids, suggesting potential for strategic price increases.
                
                **Recommended approach:**
                1. Test incremental price increases of **${impact['potential_price_increase']:.2f}** per CPI
                2. Focus on projects where the top factors indicate higher price tolerance
                3. Monitor win rates closely after implementing price changes
                4. Potential revenue impact: **${monthly_impact:.2f}** monthly
                """)
            elif price_gap_pct > 10:
                st.warning(f"""
                ### Moderate pricing opportunity identified
                
                Your won bids are priced **{price_gap_pct:.1f}%** lower than lost bids, suggesting room for selective price increases.
                
                **Recommended approach:**
                1. Test selective price increases of **${impact['potential_price_increase']:.2f}** per CPI on less price-sensitive projects
                2. Identify projects with factors that show higher price tolerance
                3. Potential revenue impact: **${monthly_impact:.2f}** monthly
                """)
            else:
                st.info(f"""
                ### Limited pricing gap detected
                
                The price difference between won and lost bids is only **{price_gap_pct:.1f}%**, suggesting current pricing is close to optimal.
                
                **Recommended approach:**
                1. Focus on efficiency improvements rather than price increases
                2. Consider small price adjustments of **${impact['potential_price_increase']:.2f}** per CPI for specific project types
                3. Look for cost optimization opportunities
                """)
    
    except Exception as e:
        logger.error(f"Error showing pricing factors: {e}", exc_info=True)
        st.error(f"Error analyzing pricing factors: {str(e)}")

def show_data_analysis(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Show comprehensive data analysis in Streamlit.
    
    Args:
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
        combined_data (pd.DataFrame): Combined DataFrame of won and lost bids
    """
    st.markdown("# Data Analysis Dashboard")
    st.markdown("## Comprehensive data quality, feature engineering, and modeling analysis")
    
    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data Quality Issues", 
        "Feature Engineering Potential", 
        "Modeling Suitability", 
        "Key Pricing Factors"
    ])
    
    with tab1:
        dataset = st.selectbox(
            "Select dataset to analyze",
            ["Combined Data", "Won Bids", "Lost Bids"]
        )
        
        if dataset == "Combined Data":
            show_data_quality_analysis(combined_data, "Combined Data")
        elif dataset == "Won Bids":
            show_data_quality_analysis(won_data, "Won Bids")
        else:
            show_data_quality_analysis(lost_data, "Lost Bids")
    
    with tab2:
        show_feature_engineering_potential(combined_data)
    
    with tab3:
        show_modeling_suitability(won_data, lost_data)
    
    with tab4:
        show_pricing_factors(combined_data)