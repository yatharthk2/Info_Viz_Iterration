"""
Data processing utilities for the CPI Analysis & Prediction Dashboard.
Handles data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load CPI data from Excel files.
    Loads both won bids (invoiced jobs) and lost bids (DealItemReportLOST).
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing different dataframes:
            - 'won': Won deals dataframe
            - 'won_filtered': Won deals with extreme values filtered out
            - 'lost': Lost deals dataframe
            - 'lost_filtered': Lost deals with extreme values filtered out
            - 'combined': Combined dataframe of won and lost deals
            - 'combined_filtered': Combined dataframe with extreme values filtered out
    """
    try:
        # Define file paths
        won_file_path = 'attached_assets/invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx'
        lost_file_path = 'attached_assets/DealItemReportLOST.xlsx'
        
        # Check if files exist
        if os.path.exists(won_file_path) and os.path.exists(lost_file_path):
            logger.info(f"Loading data from Excel files")
            
            # Load won bids (invoiced jobs)
            won_data = pd.read_excel(won_file_path)
            logger.info(f"Loaded won data: {won_data.shape[0]} rows, {won_data.shape[1]} columns")
            
            # Load lost bids (DealItemReportLOST)
            lost_data = pd.read_excel(lost_file_path)
            logger.info(f"Loaded lost data: {lost_data.shape[0]} rows, {lost_data.shape[1]} columns")
            
            # Process won data based on data dictionary
            # As per data dictionary: invoiced jobs this year
            won_processed = process_won_data(won_data)
            
            # Process lost data based on data dictionary
            # As per data dictionary: Deal Item Report Lost
            lost_processed = process_lost_data(lost_data)
            
            # Combine the data
            won_processed['Type'] = 'Won'
            lost_processed['Type'] = 'Lost'
            combined_data = pd.concat([won_processed, lost_processed], ignore_index=True)
            
            # Clean the combined data
            combined_filtered = clean_data(combined_data, filter_extremes=True)
            won_filtered = combined_filtered[combined_filtered['Type'] == 'Won']
            lost_filtered = combined_filtered[combined_filtered['Type'] == 'Lost']
            
            # Also clean individual datasets for separate analysis
            won_cleaned = clean_data(won_processed, filter_extremes=True)
            lost_cleaned = clean_data(lost_processed, filter_extremes=True)
            
            # Create dictionary of dataframes
            data_dict = {
                'won': won_processed,
                'won_filtered': won_cleaned,
                'lost': lost_processed,
                'lost_filtered': lost_cleaned,
                'combined': combined_data,
                'combined_filtered': combined_filtered
            }
            
            return data_dict
        
        # If files don't exist or can't be loaded, create a minimal dataset for testing
        logger.warning("Excel files not found. Creating minimal demo dataset.")
        
        # Create synthetic data for testing
        data = {
            'Type': ['Won', 'Won', 'Won', 'Lost', 'Lost', 'Won', 'Lost', 'Won', 'Lost', 'Won'],
            'IR': [20, 45, 10, 15, 50, 30, 25, 40, 35, 5],
            'LOI': [10, 15, 20, 15, 10, 12, 18, 8, 25, 15],
            'Completes': [200, 500, 300, 250, 400, 350, 200, 600, 300, 150],
            'CPI': [15.50, 10.25, 25.75, 30.00, 18.50, 12.75, 22.00, 9.50, 28.00, 35.00]
        }
        
        df = pd.DataFrame(data)
        filtered_df = clean_data(df, filter_extremes=True)
        
        won_data = df[df['Type'] == 'Won']
        lost_data = df[df['Type'] == 'Lost']
        won_filtered = filtered_df[filtered_df['Type'] == 'Won']
        lost_filtered = filtered_df[filtered_df['Type'] == 'Lost']
        
        return {
            'won': won_data,
            'won_filtered': won_filtered,
            'lost': lost_data,
            'lost_filtered': lost_filtered,
            'combined': df,
            'combined_filtered': filtered_df
        }
    
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        # Create a minimal dataset in case of error
        data = {
            'Type': ['Won', 'Won', 'Won', 'Lost', 'Lost'],
            'IR': [20, 45, 10, 15, 50],
            'LOI': [10, 15, 20, 15, 10],
            'Completes': [200, 500, 300, 250, 400],
            'CPI': [15.50, 10.25, 25.75, 30.00, 18.50]
        }
        
        df = pd.DataFrame(data)
        
        return {
            'won': df[df['Type'] == 'Won'],
            'won_filtered': df[df['Type'] == 'Won'],
            'lost': df[df['Type'] == 'Lost'],
            'lost_filtered': df[df['Type'] == 'Lost'],
            'combined': df,
            'combined_filtered': df
        }


def process_won_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the won bids data (invoiced jobs) based on the data dictionary.
    
    Args:
        df (pd.DataFrame): Raw dataframe of invoiced jobs
        
    Returns:
        pd.DataFrame: Processed dataframe with standardized columns
    """
    try:
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # According to data dictionary:
        # Column M = CPI
        # Column N = Actual IR
        # Column O = Actual LOI
        # Column L = Complete (Sample Size)
        
        # Map columns to standardized names
        col_mapping = {
            'M': 'CPI',
            'N': 'IR',
            'O': 'LOI',
            'L': 'Completes'
        }
        
        # Initialize result dataframe with required columns
        result_df = pd.DataFrame()
        
        # Extract and rename columns
        for excel_col, new_name in col_mapping.items():
            # Find the actual column name that starts with "Column -" + excel_col
            # If not found, try using the Excel column letter or index directly
            col_candidates = [c for c in processed_df.columns if c.startswith(f"Column - {excel_col}") or c == excel_col]
            
            if col_candidates:
                col_name = col_candidates[0]
                result_df[new_name] = processed_df[col_name]
            else:
                # Try to use column index
                try:
                    col_idx = ord(excel_col) - ord('A')
                    if 0 <= col_idx < processed_df.shape[1]:
                        result_df[new_name] = processed_df.iloc[:, col_idx]
                    else:
                        logger.warning(f"Could not map column {excel_col} to index {col_idx}")
                        result_df[new_name] = np.nan
                except Exception as e:
                    logger.warning(f"Error mapping column {excel_col}: {e}")
                    result_df[new_name] = np.nan
        
        # Ensure all required columns are present
        for col in ['CPI', 'IR', 'LOI', 'Completes']:
            if col not in result_df.columns:
                logger.warning(f"Column {col} not found in won data, using placeholder values")
                result_df[col] = np.nan
        
        # Attempt to convert columns to appropriate data types
        for col in result_df.columns:
            if col in ['CPI', 'IR', 'LOI']:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            elif col == 'Completes':
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
        
        # Filter out rows with NaN values in critical columns
        result_df = result_df.dropna(subset=['CPI'])
        
        logger.info(f"Processed won data: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
        return result_df
    
    except Exception as e:
        logger.error(f"Error processing won data: {e}", exc_info=True)
        # Return an empty dataframe with required columns
        return pd.DataFrame(columns=['CPI', 'IR', 'LOI', 'Completes'])


def process_lost_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the lost bids data (DealItemReportLOST) based on the data dictionary.
    
    Args:
        df (pd.DataFrame): Raw dataframe of lost bids
        
    Returns:
        pd.DataFrame: Processed dataframe with standardized columns
    """
    try:
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # According to data dictionary:
        # Column F = IR
        # Column G = LOI
        # Column H = CustomerRate (quoted to customer) -> This becomes our CPI for lost bids
        # Column E = Qty (the unit quantity being sold) -> This becomes our Completes
        
        # Map columns to standardized names
        col_mapping = {
            'F': 'IR',
            'G': 'LOI',
            'H': 'CPI',
            'E': 'Completes'
        }
        
        # Initialize result dataframe with required columns
        result_df = pd.DataFrame()
        
        # Extract and rename columns
        for excel_col, new_name in col_mapping.items():
            # Find the actual column name that starts with "Column -" + excel_col
            # If not found, try using the Excel column letter or index directly
            col_candidates = [c for c in processed_df.columns if c.startswith(f"Column - {excel_col}") or c == excel_col]
            
            if col_candidates:
                col_name = col_candidates[0]
                result_df[new_name] = processed_df[col_name]
            else:
                # Try to use column index
                try:
                    col_idx = ord(excel_col) - ord('A')
                    if 0 <= col_idx < processed_df.shape[1]:
                        result_df[new_name] = processed_df.iloc[:, col_idx]
                    else:
                        logger.warning(f"Could not map column {excel_col} to index {col_idx}")
                        result_df[new_name] = np.nan
                except Exception as e:
                    logger.warning(f"Error mapping column {excel_col}: {e}")
                    result_df[new_name] = np.nan
        
        # Ensure all required columns are present
        for col in ['CPI', 'IR', 'LOI', 'Completes']:
            if col not in result_df.columns:
                logger.warning(f"Column {col} not found in lost data, using placeholder values")
                result_df[col] = np.nan
        
        # Attempt to convert columns to appropriate data types
        for col in result_df.columns:
            if col in ['CPI', 'IR', 'LOI']:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
            elif col == 'Completes':
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
        
        # Filter out rows with NaN values in critical columns
        result_df = result_df.dropna(subset=['CPI'])
        
        logger.info(f"Processed lost data: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
        return result_df
    
    except Exception as e:
        logger.error(f"Error processing lost data: {e}", exc_info=True)
        # Return an empty dataframe with required columns
        return pd.DataFrame(columns=['CPI', 'IR', 'LOI', 'Completes'])

def clean_data(data: pd.DataFrame, filter_extremes: bool = True) -> pd.DataFrame:
    """
    Clean and prepare the CPI data.
    
    Args:
        data (pd.DataFrame): Raw data DataFrame
        filter_extremes (bool): Whether to filter out extreme values
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Create a copy of the data to avoid modifying the original
        cleaned_data = data.copy()
        
        # Check for required columns
        required_columns = ['Type', 'IR', 'LOI', 'Completes', 'CPI']
        for col in required_columns:
            if col not in cleaned_data.columns:
                logger.error(f"Required column '{col}' not found in data")
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Ensure consistent types
        type_mappings = {
            'Type': str,
            'IR': float,
            'LOI': float,
            'Completes': int,
            'CPI': float
        }
        
        for col, dtype in type_mappings.items():
            try:
                if col == 'Completes':
                    cleaned_data[col] = cleaned_data[col].fillna(0).astype(int)
                else:
                    cleaned_data[col] = cleaned_data[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Error converting column '{col}' to {dtype}: {e}")
                # Try to handle common issues
                if dtype == float:
                    # Try to remove non-numeric characters and convert
                    if cleaned_data[col].dtype == 'object':
                        cleaned_data[col] = pd.to_numeric(cleaned_data[col].str.replace('[^0-9.]', '', regex=True), errors='coerce')
        
        # Handle missing values
        for col in cleaned_data.columns:
            if cleaned_data[col].isnull().any():
                if col in ['IR', 'LOI', 'Completes', 'CPI']:
                    # For numeric columns, fill with median
                    median_val = cleaned_data[col].median()
                    cleaned_data[col] = cleaned_data[col].fillna(median_val)
                    logger.info(f"Filled {cleaned_data[col].isnull().sum()} missing values in '{col}' with median ({median_val})")
                else:
                    # For non-numeric columns, fill with mode
                    mode_val = cleaned_data[col].mode()[0]
                    cleaned_data[col] = cleaned_data[col].fillna(mode_val)
                    logger.info(f"Filled {cleaned_data[col].isnull().sum()} missing values in '{col}' with mode ({mode_val})")
        
        # Ensure IR is a percentage
        if cleaned_data['IR'].max() > 100:
            logger.warning("IR values greater than 100 found, scaling to percentage")
            cleaned_data['IR'] = cleaned_data['IR'] / 100
        
        # Filter out extreme values if requested
        if filter_extremes:
            # For numeric columns, filter values outside 3 standard deviations
            for col in ['IR', 'LOI', 'Completes', 'CPI']:
                mean = cleaned_data[col].mean()
                std = cleaned_data[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                # Count extreme values
                extreme_count = ((cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)).sum()
                
                if extreme_count > 0:
                    logger.info(f"Filtering {extreme_count} extreme values in '{col}'")
                    cleaned_data = cleaned_data[(cleaned_data[col] >= lower_bound) & (cleaned_data[col] <= upper_bound)]
        
        # Create IR bins for analysis
        cleaned_data['IR_Bin'] = pd.cut(
            cleaned_data['IR'],
            bins=[0, 10, 20, 30, 40, 50, 100],
            labels=['0-10%', '11-20%', '21-30%', '31-40%', '41-50%', '51-100%']
        )
        
        # Create LOI bins for analysis
        cleaned_data['LOI_Bin'] = pd.cut(
            cleaned_data['LOI'],
            bins=[0, 10, 15, 20, 30, 60],
            labels=['0-10min', '11-15min', '16-20min', '21-30min', '31-60min']
        )
        
        # Create sample size bins for analysis
        cleaned_data['Completes_Bin'] = pd.cut(
            cleaned_data['Completes'],
            bins=[0, 100, 300, 500, 1000, float('inf')],
            labels=['1-100', '101-300', '301-500', '501-1000', '1000+']
        )
        
        return cleaned_data
    
    except Exception as e:
        logger.error(f"Error cleaning data: {e}", exc_info=True)
        # Return the original data if cleaning fails
        return data

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform feature engineering for model training.
    
    Args:
        data (pd.DataFrame): Cleaned data DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    try:
        # Create a copy of the data to avoid modifying the original
        engineered_data = data.copy()
        
        # Create interaction features
        engineered_data['IR_LOI_Ratio'] = engineered_data['IR'] / engineered_data['LOI'].replace(0, 0.1)
        engineered_data['IR_Completes_Ratio'] = engineered_data['IR'] / engineered_data['Completes'].replace(0, 1)
        engineered_data['LOI_Completes_Ratio'] = engineered_data['LOI'] / engineered_data['Completes'].replace(0, 1)
        
        # Create polynomial features
        engineered_data['IR_Squared'] = engineered_data['IR'] ** 2
        engineered_data['LOI_Squared'] = engineered_data['LOI'] ** 2
        
        # Create log transforms for skewed variables
        engineered_data['Log_Completes'] = np.log1p(engineered_data['Completes'])
        
        # Create interaction terms
        engineered_data['IR_LOI_Product'] = engineered_data['IR'] * engineered_data['LOI']
        engineered_data['Log_IR_LOI_Product'] = np.log1p(engineered_data['IR_LOI_Product'])
        
        # Calculate derived metrics
        engineered_data['CPI_per_Minute'] = engineered_data['CPI'] / engineered_data['LOI'].replace(0, 0.1)
        
        # One-hot encode categorical variables
        engineered_data = pd.get_dummies(engineered_data, columns=['Type'], prefix=['Type'])
        
        # Ensure Type_Won exists (handle case where all data is one type)
        if 'Type_Won' not in engineered_data.columns:
            engineered_data['Type_Won'] = 0
            
        # Ensure Type_Lost exists
        if 'Type_Lost' not in engineered_data.columns:
            engineered_data['Type_Lost'] = 0
        
        return engineered_data
    
    except Exception as e:
        logger.error(f"Error engineering features: {e}", exc_info=True)
        # Return the original data if feature engineering fails
        return data
