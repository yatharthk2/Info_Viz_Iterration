# CPI Analysis & Prediction Dashboard - Code Documentation

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Core Modules](#core-modules)
4. [Utility Modules](#utility-modules)
5. [Component Modules](#component-modules)
6. [Model Modules](#model-modules)
7. [Configuration](#configuration)

## System Architecture

The CPI Analysis & Prediction Dashboard is a Streamlit-based web application that provides insights and predictions for Cost Per Interview (CPI) optimization in market research projects. The application follows a modular architecture:

```
                  ┌─────────────┐
                  │    app.py   │ ◄── Entry point
                  └─────┬───────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
┌────────▼────────┐ ┌───▼───┐  ┌──────▼───────┐
│ data_processor.py│ │utils/  │  │ components/ │
└────────┬────────┘ └───┬───┘  └──────┬───────┘
         │              │              │
         │         ┌────▼────┐   ┌────▼─────┐
         └────────►│ models/ │◄──┤ theme.py │
                   └─────────┘   └──────────┘
```

## Data Flow

The data flows through the system as follows:

1. **Data Loading**: Excel files are loaded from the `attached_assets` folder
   - `invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx` (won bids)
   - `DealItemReportLOST.xlsx` (lost bids)

2. **Data Processing**: Raw data is transformed into standardized formats
   - Column mapping according to the data dictionary
   - Data cleaning (missing values, outliers)
   - Feature engineering (binning, derived metrics)
   - Dataset creation (won, lost, combined)

3. **Analysis & Visualization**: Processed data is analyzed and visualized
   - Overview metrics and trends
   - Detailed analysis by factors (IR, LOI, Sample Size)
   - Insights and recommendations

4. **Prediction**: Models are trained and used for CPI prediction
   - User input collection
   - Model prediction
   - Strategy recommendation

## Core Modules

### app.py
**Purpose**: Main application entry point that orchestrates the dashboard components.

**Key Functions**:
- `main()`: Initializes the application, loads data, and handles navigation
- `show_overview()`: Displays the overview dashboard
- `show_analysis()`: Displays the detailed analysis dashboard
- `show_prediction()`: Displays the CPI prediction tool
- `show_insights()`: Displays the insights and recommendations dashboard
- `show_model_monitoring()`: Displays the model monitoring dashboard

**Data Flow**:
- Loads data using `data_processor.py`
- Applies theme using `utils/theme.py`
- Renders components based on user navigation

### data_processor.py
**Purpose**: Handles data loading, cleaning, and feature engineering.

**Key Functions**:
- `load_data()`: Loads data from Excel files and returns processed dataframes
- `process_won_data()`: Processes the won bids data based on data dictionary
- `process_lost_data()`: Processes the lost bids data based on data dictionary
- `clean_data()`: Cleans data by handling missing values and outliers
- `engineer_features()`: Adds derived features for analysis and modeling

**Data Flow**:
- Reads Excel files from the `attached_assets` folder
- Transforms raw data into standardized format
- Handles data quality issues (missing values, outliers)
- Creates derived features and bins for analysis

**Detailed Functions**:

#### `load_data()`
```python
def load_data() -> Dict[str, pd.DataFrame]:
    """
    Load CPI data from Excel files with extensive error handling and data quality checks.
    """
    logger = logging.getLogger('data_processor')
    logger.info("Loading data from Excel files")
    
    # File paths with validation
    won_path = 'attached_assets/invoiced_jobs_this_year_20240912T18_36_36.439126Z.xlsx'
    lost_path = 'attached_assets/DealItemReportLOST.xlsx'
    
    if not os.path.exists(won_path):
        raise FileNotFoundError(f"Won bids data file not found at {won_path}")
    if not os.path.exists(lost_path):
        raise FileNotFoundError(f"Lost bids data file not found at {lost_path}")
    
    # Load with error handling
    try:
        won_data_raw = pd.read_excel(won_path)
        logger.info(f"Loaded won data: {len(won_data_raw)} rows, {len(won_data_raw.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading won data: {str(e)}")
        raise ValueError(f"Failed to load won data: {str(e)}")
    
    try:
        lost_data_raw = pd.read_excel(lost_path)
        logger.info(f"Loaded lost data: {len(lost_data_raw)} rows, {len(lost_data_raw.columns)} columns")
    except Exception as e:
        logger.error(f"Error loading lost data: {str(e)}")
        raise ValueError(f"Failed to load lost data: {str(e)}")
    
    # Process the data
    won_data = process_won_data(won_data_raw)
    lost_data = process_lost_data(lost_data_raw)
    
    # Add Type field for identification
    won_data['Type'] = 'Won'
    lost_data['Type'] = 'Lost'
    
    # Data quality logging
    log_data_quality_issues(won_data, 'won_data')
    log_data_quality_issues(lost_data, 'lost_data')
    
    # Clean and filter
    won_data_filtered = clean_data(won_data, filter_extremes=True)
    lost_data_filtered = clean_data(lost_data, filter_extremes=True)
    
    # Combine datasets
    combined_data = pd.concat([won_data, lost_data], ignore_index=True)
    combined_data_filtered = pd.concat([won_data_filtered, lost_data_filtered], ignore_index=True)
    
    # Apply binning
    combined_data = apply_all_bins(combined_data)
    combined_data_filtered = apply_all_bins(combined_data_filtered)
    won_data = apply_all_bins(won_data)
    won_data_filtered = apply_all_bins(won_data_filtered)
    lost_data = apply_all_bins(lost_data)
    lost_data_filtered = apply_all_bins(lost_data_filtered)
    
    return {
        'won': won_data,
        'won_filtered': won_data_filtered,
        'lost': lost_data,
        'lost_filtered': lost_data_filtered,
        'combined': combined_data,
        'combined_filtered': combined_data_filtered
    }
```

#### `process_won_data()`
```python
def process_won_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the won bids data using column mapping from data dictionary.
    """
    # Extract relevant columns with advanced error handling
    try:
        # Column name mapping based on data dictionary
        column_mapping = {
            'ProjectId': 'ID',
            'InvoicedJobId': 'Job_ID',
            'ProjectName': 'Name',
            'NumberOfCompletes': 'Completes',
            'LengthOfInterview': 'LOI',
            'IncidenceRate': 'IR',
            'RevenueAmountPerComplete': 'CPI',
            'Country': 'Country'
        }
        
        # Select and rename columns
        required_columns = list(column_mapping.keys())
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in won data: {', '.join(missing_columns)}")
        
        result_df = df[required_columns].copy()
        result_df.rename(columns=column_mapping, inplace=True)
        
        # Data conversions with validation
        numeric_columns = ['Completes', 'LOI', 'IR', 'CPI']
        for col in numeric_columns:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        # Filter out rows with null values in critical columns
        critical_columns = ['CPI', 'Completes']
        initial_count = len(result_df)
        result_df = result_df.dropna(subset=critical_columns)
        dropped_count = initial_count - len(result_df)
        
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with null values in critical columns")
        
        # Data validation checks
        if (result_df['CPI'] <= 0).any():
            logger.warning(f"Found {(result_df['CPI'] <= 0).sum()} rows with zero or negative CPI")
            result_df = result_df[result_df['CPI'] > 0]
        
        logger.info(f"Processed won data: {len(result_df)} rows, {len(result_df.columns)} columns")
        return result_df
        
    except Exception as e:
        logger.error(f"Error processing won data: {str(e)}")
        raise ValueError(f"Failed to process won data: {str(e)}")
```

#### `clean_data()`
```python
def clean_data(data: pd.DataFrame, filter_extremes: bool = True) -> pd.DataFrame:
    """
    Clean and prepare the CPI data with advanced data quality handling.
    """
    df = data.copy()
    
    # Fill missing values with medians
    for col in ['IR', 'LOI']:
        if col in df.columns:
            median_value = df[col].median()
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Filled {missing_count} missing values in '{col}' with median ({median_value})")
                df[col] = df[col].fillna(median_value)
    
    # Convert IR to percentage if needed
    if 'IR' in df.columns and (df['IR'] > 100).any():
        logger.warning("IR values greater than 100 found, scaling to percentage")
        mask = df['IR'] > 100
        df.loc[mask, 'IR'] = df.loc[mask, 'IR'] / 100
    
    # Filter extreme values if requested
    if filter_extremes:
        for col in ['IR', 'LOI', 'Completes', 'CPI']:
            if col in df.columns:
                initial_count = len(df)
                df = cap_outliers(df, col, quantile=0.95)
                filtered_count = initial_count - len(df)
                logger.info(f"Filtering {filtered_count} extreme values in '{col}'")
    
    return df
```

#### `engineer_features()`
```python
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform advanced feature engineering for model training.
    """
    df = data.copy()
    
    # Create interaction features
    df['IR_LOI'] = df['IR'] * df['LOI']
    df['CPI_per_minute'] = df['CPI'] / df['LOI']
    df['Completes_Log'] = np.log1p(df['Completes'])
    
    # Create country-related features
    if 'Country' in df.columns:
        df['Country_Group'] = df['Country'].apply(categorize_country)
        
        # One-hot encode country groups
        country_dummies = pd.get_dummies(df['Country_Group'], prefix='Country')
        df = pd.concat([df, country_dummies], axis=1)
    
    # Create bin-related features
    if 'IR_Bin' in df.columns and 'LOI_Bin' in df.columns:
        df['IR_LOI_Segment'] = df['IR_Bin'] + '_' + df['LOI_Bin']
        
    # Feature scaling for modeling
    numeric_features = ['IR', 'LOI', 'Completes', 'IR_LOI', 'CPI_per_minute', 'Completes_Log']
    for feature in numeric_features:
        if feature in df.columns:
            scaler = StandardScaler()
            df[f'{feature}_scaled'] = scaler.fit_transform(df[[feature]])
    
    return df
```

## Utility Modules

### utils/theme.py
**Purpose**: Defines the visual styling for the application.

**Key Components**:
- `COLOR_SYSTEM`: Dictionary of color palettes for different elements
- `TYPOGRAPHY`: Typography specifications for consistent text styling
- `apply_theme()`: Applies the custom dark high-contrast theme to the app
- `format_chart_for_dark_mode()`: Formats Plotly charts for dark mode

**Detailed Structure**:
```python
# Color system definition
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
        'DARKEST': '#121526',    # Very dark blue - backgrounds
        'DARKER': '#1E2538',     # Dark blue - cards, panels
        'DARK': '#2E3447',       # Medium dark - borders, dividers
        'MEDIUM': '#4A4F67',     # Medium - secondary UI elements
        'LIGHT': '#9FA3B8',      # Light - secondary text
        'LIGHTER': '#CFD1DC'     # Very light - disabled states
    },
    
    # Semantic colors
    'SEMANTIC': {
        'SUCCESS': '#52BC9F',    # Green - success states
        'WARNING': '#F6C85F',    # Yellow - warning states
        'ERROR': '#E15759',      # Red - error states
        'INFO': '#4e79a7'        # Blue - info states
    },
    
    # Chart colors - optimized for data visualization
    'CHARTS': {
        'WON': '#4e79a7',        # Blue for won bids
        'LOST': '#f28e2b',       # Orange for lost bids
        'SERIES1': '#4e79a7',    # Blue
        'SERIES2': '#f28e2b',    # Orange
        'SERIES3': '#59a14f',    # Green
        'SERIES4': '#e15759',    # Red
        'SERIES5': '#b07aa1',    # Purple
        'SERIES6': '#9d7660',    # Brown
        'SERIES7': '#76b7b2',    # Teal
        'SERIES8': '#ff9da7'     # Pink
    }
}
```

### utils/visualization.py
**Purpose**: Provides plotting functions for data visualization.

**Key Functions**:
- `create_cpi_comparison_chart()`: Creates a bar chart comparing CPI by bid type
- `create_cpi_factor_chart()`: Creates a scatter plot of CPI vs. a given factor
- `create_win_rate_chart()`: Creates a bar chart showing win rates by factor bins
- `create_distribution_chart()`: Creates histograms of data distributions
- `create_correlation_heatmap()`: Creates a correlation matrix heatmap

### utils/data_quality.py
**Purpose**: Analyzes and reports on data quality issues.

**Key Functions**:
- `analyze_data_quality()`: Performs comprehensive data quality analysis
- `detect_missing_values()`: Detects and quantifies missing values
- `detect_outliers()`: Identifies outliers using various methods
- `analyze_distributions()`: Analyzes data distributions for skewness
- `create_data_quality_report()`: Generates a data quality report with metrics

### utils/quick_insights.py
**Purpose**: Generates concise, emoji-powered summaries of key data trends.

**Key Functions**:
- `calculate_cpi_diff_percentage()`: Calculates percentage difference in CPI
- `get_correlations()`: Finds correlations between CPI and other variables
- `identify_key_segments()`: Identifies segments with highest win rates
- `analyze_outliers()`: Quantifies outliers across key variables
- `generate_quick_insights()`: Creates a markdown summary of key insights

**Detailed Functions**:

#### `generate_quick_insights()`
```python
def generate_quick_insights(won_data: pd.DataFrame, lost_data: pd.DataFrame, 
                         combined_data: pd.DataFrame) -> str:
    """
    Generate a concise, emoji-powered summary of key data trends.
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
```

### utils/model_monitoring.py
**Purpose**: Tracks model performance metrics over time to detect drift.

**Key Functions**:
- `track_prediction()`: Records prediction and actual values for tracking
- `calculate_performance_metrics()`: Computes model performance metrics
- `detect_model_drift()`: Detects significant changes in model performance
- `create_metrics_trend_chart()`: Visualizes performance metric trends
- `show_model_monitoring()`: Displays the model monitoring dashboard

## Component Modules

### components/overview.py
**Purpose**: Displays a high-level summary of the data and key metrics.

**Key Functions**:
- `show_overview()`: Displays the overview dashboard
- `show_key_metrics()`: Shows summary metrics for won and lost bids
- `show_cpi_comparison()`: Compares CPI between won and lost bids
- `show_factor_charts()`: Shows CPI vs. key factors

**Detailed Functions**:

#### `show_overview()`
```python
def show_overview(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Display the overview dashboard with comprehensive metrics and interactive visualizations.
    """
    st.title("CPI Analysis Overview")
    
    st.markdown("""
    This dashboard provides analysis of Cost Per Interview (CPI) between won and lost bids
    to optimize pricing strategies for future bids.
    """)
    
    # Key Metrics Section
    st.header("Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Won bids metrics
    with col1:
        st.metric(
            label="Won Bids", 
            value=f"{len(won_data):,}",
            delta=f"{(len(won_data) / (len(won_data) + len(lost_data)) * 100):.1f}%",
            delta_color="normal",
            help="Number of won bids and percentage of total bids"
        )
    
    # Lost bids metrics
    with col2:
        st.metric(
            label="Lost Bids", 
            value=f"{len(lost_data):,}",
            delta=None,
            help="Number of lost bids"
        )
    
    # Median CPI comparison
    with col3:
        won_median = won_data['CPI'].median()
        lost_median = lost_data['CPI'].median()
        delta_pct = ((won_median - lost_median) / lost_median) * 100
        
        st.metric(
            label="Median CPI (Won)", 
            value=f"${won_median:.2f}",
            delta=f"{delta_pct:.1f}% vs Lost",
            delta_color="inverse", # Lower is better for CPI
            help="Median Cost Per Interview for won bids compared to lost bids"
        )
    
    # Win rate by IR
    with col4:
        win_rates = combined_data.groupby('IR_Bin')['Type'].apply(
            lambda x: (x == 'Won').mean() * 100
        )
        best_ir = win_rates.idxmax()
        best_rate = win_rates.max()
        
        st.metric(
            label="Best IR Segment", 
            value=f"{best_ir}",
            delta=f"{best_rate:.1f}% Win Rate",
            delta_color="normal",
            help="IR segment with the highest win rate"
        )
    
    # CPI Comparison Chart
    st.header("CPI Comparison")
    
    tab1, tab2 = st.tabs(["💰 CPI by Bid Outcome", "📊 CPI Distribution"])
    
    with tab1:
        chart = create_cpi_comparison_chart(won_data, lost_data)
        st.plotly_chart(chart, use_container_width=True)
        
        with st.expander("Analysis Insights"):
            st.markdown("""
            * Won bids typically have lower CPIs compared to lost bids
            * The difference is more pronounced in higher IR segments
            * Consider this gap when pricing new bids
            """)
    
    with tab2:
        chart = create_distribution_chart(won_data, lost_data, 'CPI')
        st.plotly_chart(chart, use_container_width=True)
        
        with st.expander("Distribution Insights"):
            st.markdown("""
            * Won bids show a narrower distribution, clustered around the median
            * Lost bids have a wider spread, with more instances of higher CPIs
            * The overlap area represents the competitive pricing zone
            """)
    
    # CPI vs Key Factors
    st.header("CPI vs Key Factors")
    
    factor_options = [
        ("Incidence Rate (IR)", "IR"),
        ("Length of Interview (LOI)", "LOI"),
        ("Sample Size (Completes)", "Completes")
    ]
    
    selected_factor = st.selectbox(
        "Select Factor to Analyze",
        options=[option[0] for option in factor_options],
        index=0
    )
    
    # Get the column name for the selected factor
    factor_col = next(option[1] for option in factor_options if option[0] == selected_factor)
    
    # Create and display the factor chart
    chart = create_cpi_factor_chart(combined_data, factor_col)
    st.plotly_chart(chart, use_container_width=True)
    
    # Dynamic insights based on selected factor
    with st.expander(f"CPI vs {selected_factor} Insights"):
        if factor_col == "IR":
            st.markdown("""
            * CPI tends to increase as IR decreases, showing a clear inverse relationship
            * Lost bids often have higher CPIs at the same IR level
            * The optimal pricing strategy should account for this relationship
            """)
        elif factor_col == "LOI":
            st.markdown("""
            * CPI generally increases with longer interview lengths
            * The relationship is mostly linear for won bids
            * Consider a pricing formula based on LOI for consistent bidding
            """)
        else:  # Completes
            st.markdown("""
            * Larger sample sizes often have lower CPIs (economies of scale)
            * The discount for volume is more evident in won bids
            * Consider volume discounts in pricing strategy
            """)
    
    # Win Rate Analysis
    st.header("Win Rate Analysis")
    
    win_rate_factor = st.radio(
        "Analyze Win Rate By:",
        options=["IR_Bin", "LOI_Bin", "Completes_Bin"],
        format_func=lambda x: x.split("_")[0],
        horizontal=True
    )
    
    chart = create_win_rate_chart(combined_data, win_rate_factor)
    st.plotly_chart(chart, use_container_width=True)
    
    # Quick Insights Button
    show_quick_insights(won_data, lost_data, combined_data)
```

### components/analysis.py
**Purpose**: Provides detailed analysis of how different factors affect CPI and win rates.

**Key Functions**:
- `show_analysis()`: Displays the analysis dashboard
- `show_factor_deep_dive()`: Shows detailed analysis of a factor's effect on CPI
- `show_interaction_analysis()`: Analyzes interactions between factors
- `show_win_rate_analysis()`: Analyzes win rates by different factors
- `show_data_quality_analysis()`: Shows data quality metrics and issues

**Detailed Functions**:

#### `show_factor_deep_dive()`
```python
def show_factor_deep_dive(data: pd.DataFrame, factor: str, factor_bin: str, 
                         won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Provide detailed analysis of a specific factor's impact on CPI.
    """
    # Factor metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        corr = data[['CPI', factor]].corr().iloc[0, 1]
        st.metric(
            label=f"Correlation with CPI", 
            value=f"{corr:.2f}",
            delta=None,
            help=f"Pearson correlation between {factor} and CPI"
        )
    
    with col2:
        # Win rate variation by factor
        segment_win_rates = data.groupby(factor_bin)['Type'].apply(
            lambda x: (x == 'Won').mean()
        )
        variation = segment_win_rates.max() - segment_win_rates.min()
        
        st.metric(
            label=f"Win Rate Variation", 
            value=f"{variation:.2f}",
            delta=None,
            help=f"Difference between highest and lowest win rates across {factor} segments"
        )
    
    with col3:
        # Pricing premium/discount analysis
        factor_segments = data[factor_bin].unique()
        
        if len(factor_segments) >= 2:
            # Sort segments to ensure consistent comparison
            factor_segments = sorted(factor_segments)
            
            low_segment = factor_segments[0]
            high_segment = factor_segments[-1]
            
            low_cpi = data[data[factor_bin] == low_segment]['CPI'].median()
            high_cpi = data[data[factor_bin] == high_segment]['CPI'].median()
            
            premium_pct = ((high_cpi - low_cpi) / low_cpi) * 100
            
            st.metric(
                label=f"Premium: {high_segment} vs {low_segment}", 
                value=f"{premium_pct:.1f}%",
                delta=None,
                help=f"Percentage difference in median CPI between highest and lowest {factor} segments"
            )
    
    # CPI by factor chart
    st.subheader(f"CPI by {factor} Segment")
    
    chart = create_segment_cpi_chart(data, factor_bin)
    st.plotly_chart(chart, use_container_width=True)
    
    # Won vs Lost comparison by factor
    st.subheader(f"Won vs Lost CPI by {factor}")
    
    chart = create_won_lost_comparison_chart(won_data, lost_data, factor)
    st.plotly_chart(chart, use_container_width=True)
    
    # Win rate by factor
    st.subheader(f"Win Rate by {factor}")
    
    chart = create_win_rate_chart(data, factor_bin)
    st.plotly_chart(chart, use_container_width=True)
    
    # Factor distribution
    st.subheader(f"{factor} Distribution")
    
    chart = create_distribution_chart(won_data, lost_data, factor)
    st.plotly_chart(chart, use_container_width=True)
    
    # Key insights
    with st.expander(f"Key Insights for {factor}"):
        # Dynamic insights based on factor
        if factor == 'IR':
            st.markdown("""
            * **Pricing Impact**: Lower IR projects typically require higher CPIs to account for screening costs
            * **Win Rate Pattern**: Win rates often vary significantly across IR segments
            * **Strategic Consideration**: Consider a tiered pricing approach for different IR levels
            """)
        elif factor == 'LOI':
            st.markdown("""
            * **Pricing Impact**: CPI generally increases with LOI, but not always proportionally
            * **Win Rate Pattern**: Shorter interviews may have different competitive dynamics
            * **Strategic Consideration**: Consider a base price plus per-minute rate approach
            """)
        else:  # Completes
            st.markdown("""
            * **Pricing Impact**: Larger sample sizes often benefit from economies of scale
            * **Win Rate Pattern**: Very small and very large samples might have distinct win rate patterns
            * **Strategic Consideration**: Volume discounts should be structured carefully
            """)
```

### components/prediction.py
**Purpose**: Provides an interactive tool for predicting optimal CPI for new bids.

**Key Functions**:
- `show_prediction()`: Displays the prediction tool
- `collect_user_input()`: Collects user inputs for prediction
- `show_prediction_results()`: Shows prediction results and recommendations
- `show_similar_projects()`: Shows similar past projects for comparison
- `show_win_probability()`: Shows estimated win probability at different CPI levels

**Detailed Functions**:

#### `show_prediction()`
```python
def show_prediction(combined_data_engineered: pd.DataFrame, won_data: pd.DataFrame, lost_data: pd.DataFrame) -> None:
    """
    Display the CPI prediction tool with comprehensive insights and explanations.
    """
    st.title("CPI Prediction Tool")
    
    st.markdown("""
    This tool helps estimate optimal Cost Per Interview (CPI) pricing for new bids
    based on historical data patterns. Enter project specifications to get a prediction.
    """)
    
    # Train models if not already in session state
    if 'models' not in st.session_state:
        with st.spinner("Training prediction models..."):
            X, y = prepare_model_data(combined_data_engineered)
            st.session_state.models = train_models(X, y)
            st.session_state.feature_names = X.columns.tolist()
    
    # Project Specifications Input
    st.header("Project Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ir = st.slider(
            "Incidence Rate (%)",
            min_value=1,
            max_value=100,
            value=30,
            step=1,
            help="Expected incidence rate for the project"
        )
        
        loi = st.slider(
            "Length of Interview (minutes)",
            min_value=1,
            max_value=60,
            value=15,
            step=1,
            help="Expected interview duration in minutes"
        )
    
    with col2:
        completes = st.slider(
            "Sample Size (completes)",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Required number of completed interviews"
        )
        
        country_group = st.selectbox(
            "Country Group",
            options=["North America", "Western Europe", "Eastern Europe", "Asia", "Rest of World"],
            index=0,
            help="Target geography for the project"
        )
    
    # Additional Specifications (Optional)
    with st.expander("Additional Specifications (Optional)"):
        col1, col2 = st.columns(2)
        
        with col1:
            b2b_project = st.checkbox(
                "B2B Project",
                value=False,
                help="Is this a business-to-business project?"
            )
            
            complex_quotas = st.checkbox(
                "Complex Quotas",
                value=False,
                help="Does the project have complex quota requirements?"
            )
        
        with col2:
            competitive_bid = st.checkbox(
                "Highly Competitive Bid",
                value=False,
                help="Is this a highly competitive bidding situation?"
            )
            
            strategic_client = st.checkbox(
                "Strategic Client",
                value=False,
                help="Is this for a strategic or high-value client?"
            )
    
    # Process user input
    user_input = process_user_input(
        ir=ir/100,  # Convert to decimal
        loi=loi,
        completes=completes,
        country_group=country_group,
        b2b_project=b2b_project,
        complex_quotas=complex_quotas,
        competitive_bid=competitive_bid,
        strategic_client=strategic_client
    )
    
    # Make prediction
    if st.button("📊 Predict Optimal CPI", use_container_width=True):
        with st.spinner("Analyzing data and generating prediction..."):
            # Add a slight delay for UX
            time.sleep(0.8)
            
            # Make prediction
            predictions = predict_cpi(
                st.session_state.models,
                user_input,
                st.session_state.feature_names
            )
            
            # Get metrics
            metrics = get_prediction_metrics(predictions)
            
            # Get recommendation
            won_avg = won_data['CPI'].median()
            lost_avg = lost_data['CPI'].median()
            recommendation = get_recommendation(metrics['median'], won_avg, lost_avg)
            
            # Calculate win probability
            win_prob = simulate_win_probability(
                metrics['median'],
                user_input,
                won_data,
                lost_data
            )
            
            # Display results
            show_prediction_results(metrics, recommendation, win_prob, predictions)
            
            # Show similar projects
            show_similar_projects(
                won_data,
                lost_data,
                ir=ir/100,
                loi=loi,
                completes=completes
            )
            
            # Show detailed pricing strategy
            detailed_strategy = get_detailed_pricing_strategy(
                metrics['median'],
                user_input,
                won_data,
                lost_data
            )
            
            with st.expander("📝 Detailed Pricing Strategy"):
                st.markdown(detailed_strategy)
            
            # Track prediction for model monitoring
            track_prediction(
                prediction=metrics['median'],
                inputs=user_input,
                timestamp=datetime.now()
            )
```

### components/insights.py
**Purpose**: Provides strategic recommendations and insights based on the CPI analysis.

**Key Functions**:
- `show_insights()`: Displays the insights dashboard
- `show_strategic_recommendations()`: Shows high-level strategic guidance
- `show_segment_opportunities()`: Identifies promising market segments
- `show_pricing_strategies()`: Recommends pricing strategies by project type
- `show_data_driven_narrative()`: Presents a data-driven story of findings

**Detailed Functions**:

#### `show_strategic_recommendations()`
```python
def show_strategic_recommendations(won_data: pd.DataFrame, lost_data: pd.DataFrame, combined_data: pd.DataFrame) -> None:
    """
    Generate and display strategic recommendations based on data analysis.
    """
    # Calculate key metrics
    won_median_cpi = won_data['CPI'].median()
    lost_median_cpi = lost_data['CPI'].median()
    cpi_diff_pct = ((won_median_cpi - lost_median_cpi) / lost_median_cpi) * 100
    
    # Determine if won bids tend to be higher or lower than lost bids
    pricing_direction = "lower" if cpi_diff_pct < 0 else "higher"
    
    # Calculate correlation between IR and CPI
    ir_corr = combined_data[['CPI', 'IR']].corr().iloc[0, 1]
    
    # IR pricing strategy
    if ir_corr < -0.3:
        ir_strategy = "strong inverse relationship (lower IR requires higher CPI)"
    elif ir_corr < 0:
        ir_strategy = "weak inverse relationship"
    elif ir_corr > 0.3:
        ir_strategy = "strong positive relationship (higher IR requires higher CPI)"
    else:
        ir_strategy = "weak positive relationship"
    
    # Calculate correlation between LOI and CPI
    loi_corr = combined_data[['CPI', 'LOI']].corr().iloc[0, 1]
    
    # LOI pricing strategy
    if loi_corr > 0.5:
        loi_strategy = "strong positive relationship (longer LOI justifies higher CPI)"
    elif loi_corr > 0:
        loi_strategy = "weak positive relationship"
    else:
        loi_strategy = "no clear relationship"
    
    # Calculate correlation between Completes and CPI
    completes_corr = combined_data[['CPI', 'Completes']].corr().iloc[0, 1]
    
    # Sample size pricing strategy
    if completes_corr < -0.3:
        completes_strategy = "strong inverse relationship (volume discounts work well)"
    elif completes_corr < 0:
        completes_strategy = "weak inverse relationship (modest volume discounts)"
    else:
        completes_strategy = "no clear volume discount pattern"
    
    # Win rate by type patterns
    ir_win_rates = combined_data.groupby('IR_Bin')['Type'].apply(
        lambda x: (x == 'Won').mean() * 100
    ).to_dict()
    
    loi_win_rates = combined_data.groupby('LOI_Bin')['Type'].apply(
        lambda x: (x == 'Won').mean() * 100
    ).to_dict()
    
    # Generate recommendations
    st.markdown("""
    <style>
    .recommendation-card {
        background-color: rgba(49, 51, 63, 0.7);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid rgba(0, 180, 216, 0.8);
    }
    .recommendation-title {
        color: white;
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # General pricing strategy
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">🎯 General Pricing Strategy</div>
        <p>Data shows that successful bids typically have <strong>{pricing_direction}</strong> CPIs than lost bids
        by approximately <strong>{abs(cpi_diff_pct):.1f}%</strong>. Consider this gap when pricing new bids.</p>
        
        <p>Key action items:</p>
        <ul>
            <li>Target CPIs around <strong>${won_median_cpi:.2f}</strong> as a baseline</li>
            <li>For highly competitive bids, consider pricing up to <strong>{abs(cpi_diff_pct):.1f}%</strong> {pricing_direction} than standard</li>
            <li>Develop tiered pricing models based on project characteristics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Factor-based pricing
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">📊 Factor-Based Pricing Framework</div>
        <p>Our analysis shows the following relationships between project factors and optimal CPIs:</p>
        
        <ul>
            <li><strong>Incidence Rate:</strong> {ir_strategy}</li>
            <li><strong>Length of Interview:</strong> {loi_strategy}</li>
            <li><strong>Sample Size:</strong> {completes_strategy}</li>
        </ul>
        
        <p>Recommended pricing formula approach:</p>
        <ul>
            <li>Base CPI of <strong>${won_median_cpi:.2f}</strong></li>
            <li>IR adjustment factor based on the inverse relationship</li>
            <li>LOI adjustment of approximately <strong>${(loi_corr * won_median_cpi / 15):.2f}</strong> per minute</li>
            <li>Volume discount of <strong>{min(abs(completes_corr * 100), 15):.1f}%</strong> for large projects</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Segment targeting
    # Find best and worst performing segments
    best_ir = max(ir_win_rates.items(), key=lambda x: x[1])
    worst_ir = min(ir_win_rates.items(), key=lambda x: x[1])
    
    best_loi = max(loi_win_rates.items(), key=lambda x: x[1])
    worst_loi = min(loi_win_rates.items(), key=lambda x: x[1])
    
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">🎯 Segment Targeting Strategy</div>
        <p>Focus on high-win-rate segments to maximize success:</p>
        
        <ul>
            <li>Prioritize projects with <strong>{best_ir[0]}</strong> IR (win rate: <strong>{best_ir[1]:.1f}%</strong>)</li>
            <li>Target <strong>{best_loi[0]}</strong> LOI projects (win rate: <strong>{best_loi[1]:.1f}%</strong>)</li>
            <li>Be cautious with <strong>{worst_ir[0]}</strong> IR projects (win rate only <strong>{worst_ir[1]:.1f}%</strong>)</li>
        </ul>
        
        <p>For challenging segments, adjust pricing strategy:</p>
        <ul>
            <li>For <strong>{worst_ir[0]}</strong> IR projects, consider <strong>{abs((worst_ir[1] - best_ir[1]) / 2):.1f}%</strong> lower CPIs to improve competitiveness</li>
            <li>For <strong>{worst_loi[0]}</strong> LOI projects, focus on value-add services to differentiate rather than price</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Testing and learning strategy
    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">📈 Testing & Learning Strategy</div>
        <p>Implement a structured approach to optimize pricing over time:</p>
        
        <ul>
            <li>Test different pricing points within <strong>±{abs(cpi_diff_pct):.1f}%</strong> of the recommended CPI</li>
            <li>Track win rates by segment and pricing level to identify optimal points</li>
            <li>Implement A/B testing on similar projects with different pricing strategies</li>
            <li>Regularly update the prediction model with new bid data</li>
            <li>Monitor model drift metrics to ensure prediction accuracy over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
```

## Model Modules

### models/predictor.py
**Purpose**: Makes predictions on optimal CPI based on project characteristics.

**Key Functions**:
- `predict_cpi()`: Predicts CPI based on user input
- `get_prediction_metrics()`: Calculates metrics for multiple model predictions
- `get_recommendation()`: Generates pricing recommendations
- `simulate_win_probability()`: Simulates win probability at different CPI levels

**Detailed Functions**:

#### `predict_cpi()`
```python
def predict_cpi(models: Dict[str, Any], user_input: Dict[str, float], feature_names: List[str]) -> Dict[str, float]:
    """
    Predict CPI based on user input using multiple models for robust predictions.
    
    Args:
        models (Dict[str, Any]): Dictionary of trained models
        user_input (Dict[str, float]): Dictionary of user input values
        feature_names (List[str]): List of feature names expected by the models
    
    Returns:
        Dict[str, float]: Dictionary of model predictions
    """
    # Prepare input for prediction
    X_pred = {}
    
    # For each feature required by the models
    for feature in feature_names:
        # If feature exists in user input, use it
        if feature in user_input:
            X_pred[feature] = user_input[feature]
        # Otherwise, set to 0
        else:
            X_pred[feature] = 0.0
    
    # Convert to DataFrame with correct order
    X_pred_df = pd.DataFrame([X_pred], columns=feature_names)
    
    # Make predictions with each model
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Get raw prediction
            pred = model.predict(X_pred_df)[0]
            
            # Apply additional factors from user input if available
            if 'B2B_Factor' in user_input:
                pred *= user_input['B2B_Factor']
            if 'Quota_Factor' in user_input:
                pred *= user_input['Quota_Factor']
            if 'Competitive_Factor' in user_input:
                pred *= user_input['Competitive_Factor']
            if 'Strategic_Factor' in user_input:
                pred *= user_input['Strategic_Factor']
            
            # Round to 2 decimal places
            predictions[model_name] = round(pred, 2)
        except Exception as e:
            # If model fails, use fallback
            logger.error(f"Error with {model_name} prediction: {str(e)}")
            predictions[model_name] = get_fallback_prediction(user_input, won_data, lost_data)
    
    return predictions
```

#### `get_detailed_pricing_strategy()`
```python
def get_detailed_pricing_strategy(predicted_cpi: float, user_input: Dict[str, float],
                               won_data: pd.DataFrame, lost_data: pd.DataFrame) -> str:
    """
    Generate a detailed pricing strategy based on predictions and user input.
    
    Args:
        predicted_cpi (float): Predicted CPI value
        user_input (Dict[str, float]): Dictionary of user input values
        won_data (pd.DataFrame): DataFrame of Won bids
        lost_data (pd.DataFrame): DataFrame of Lost bids
    
    Returns:
        str: Detailed pricing strategy text
    """
    # Get key metrics
    ir = user_input.get('IR', 0) * 100  # Convert back to percentage
    loi = user_input.get('LOI', 0)
    completes = user_input.get('Completes', 0)
    
    # Get similar won projects
    ir_range = (ir * 0.7, ir * 1.3)
    loi_range = (max(1, loi - 5), loi + 5)
    
    similar_won = won_data[
        (won_data['IR'] >= ir_range[0]) & 
        (won_data['IR'] <= ir_range[1]) &
        (won_data['LOI'] >= loi_range[0]) & 
        (won_data['LOI'] <= loi_range[1])
    ]
    
    similar_lost = lost_data[
        (lost_data['IR'] >= ir_range[0]) & 
        (lost_data['IR'] <= ir_range[1]) &
        (lost_data['LOI'] >= loi_range[0]) & 
        (lost_data['LOI'] <= loi_range[1])
    ]
    
    # Calculate metrics
    similar_won_count = len(similar_won)
    similar_lost_count = len(similar_lost)
    
    if similar_won_count > 0:
        similar_won_median = similar_won['CPI'].median()
        similar_won_min = similar_won['CPI'].min()
        similar_won_max = similar_won['CPI'].max()
    else:
        similar_won_median = predicted_cpi
        similar_won_min = predicted_cpi * 0.9
        similar_won_max = predicted_cpi * 1.1
    
    if similar_lost_count > 0:
        similar_lost_median = similar_lost['CPI'].median()
    else:
        similar_lost_median = predicted_cpi * 1.2
    
    # Calculate optimal price points
    aggressive_price = round(min(similar_won_median * 0.95, predicted_cpi * 0.97), 2)
    standard_price = round(predicted_cpi, 2)
    premium_price = round(max(similar_won_median * 1.05, predicted_cpi * 1.03), 2)
    
    # Calculate project total value
    total_value = round(standard_price * completes, 2)
    
    # Generate strategy text
    strategy = f"""
    ## Detailed Pricing Strategy
    
    ### Project Specifications
    - **Incidence Rate:** {ir:.1f}%
    - **Length of Interview:** {loi} minutes
    - **Sample Size:** {completes} completes
    - **Predicted CPI:** ${standard_price}
    - **Total Project Value:** ${total_value:,.2f}
    
    ### Historical Context
    - **Similar Won Projects:** {similar_won_count}
    - **Similar Lost Projects:** {similar_lost_count}
    - **Won CPI Range:** ${similar_won_min:.2f} - ${similar_won_max:.2f} (median: ${similar_won_median:.2f})
    - **Lost CPI Median:** ${similar_lost_median:.2f}
    
    ### Pricing Options
    1. **Aggressive Pricing:** ${aggressive_price}
       - Higher win probability but lower margin
       - Recommended for strategic clients or competitive situations
       - Total value: ${aggressive_price * completes:,.2f}
    
    2. **Standard Pricing:** ${standard_price}
       - Balanced approach with good win probability
       - Recommended for typical projects
       - Total value: ${standard_price * completes:,.2f}
    
    3. **Premium Pricing:** ${premium_price}
       - Higher margin but lower win probability
       - Recommended when capacity is constrained
       - Total value: ${premium_price * completes:,.2f}
    
    ### Key Considerations
    - IR is a critical factor: {get_ir_guidance(ir)}
    - LOI impact: {get_loi_guidance(loi)}
    - Sample size strategy: {get_completes_guidance(completes)}
    """
    
    # Add special factors if they exist in user input
    special_factors = []
    
    if user_input.get('B2B_Factor', 1.0) > 1.0:
        special_factors.append("- **B2B Premium:** Consider a higher price point due to B2B audience complexity")
    
    if user_input.get('Quota_Factor', 1.0) > 1.0:
        special_factors.append("- **Complex Quotas:** The complex quotas justify a pricing premium")
    
    if user_input.get('Competitive_Factor', 1.0) < 1.0:
        special_factors.append("- **Competitive Situation:** Consider more aggressive pricing to win this competitive bid")
    
    if user_input.get('Strategic_Factor', 1.0) < 1.0:
        special_factors.append("- **Strategic Client:** This strategic client relationship may justify a more competitive price")
    
    if special_factors:
        strategy += "\n\n### Special Factors\n" + "\n".join(special_factors)
    
    return strategy
```

### models/trainer.py
**Purpose**: Trains machine learning models for CPI prediction.

**Key Functions**:
- `train_models()`: Trains multiple model types for CPI prediction
- `evaluate_models()`: Evaluates model performance using cross-validation
- `select_best_model()`: Selects the best performing model
- `save_models()`: Saves trained models for later use

**Detailed Functions**:

#### `train_models()`
```python
def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Train multiple models for CPI prediction to ensure robust predictions.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (CPI)
        
    Returns:
        Dict[str, Any]: Dictionary of trained models
    """
    logger = logging.getLogger('trainer')
    logger.info(f"Training models with {len(X)} samples and {len(X.columns)} features")
    
    # Initialize model dictionary
    models = {}
    
    # 1. Linear Regression
    try:
        lr = LinearRegression()
        lr.fit(X, y)
        models['Linear Regression'] = lr
        logger.info("Trained Linear Regression model successfully")
    except Exception as e:
        logger.error(f"Error training Linear Regression: {str(e)}")
        models['Linear Regression'] = DummyModel(y.mean())
    
    # 2. Ridge Regression
    try:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        models['Ridge Regression'] = ridge
        logger.info("Trained Ridge Regression model successfully")
    except Exception as e:
        logger.error(f"Error training Ridge Regression: {str(e)}")
        models['Ridge Regression'] = DummyModel(y.mean())
    
    # 3. Random Forest Regression
    try:
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(X, y)
        models['Random Forest'] = rf
        logger.info("Trained Random Forest model successfully")
    except Exception as e:
        logger.error(f"Error training Random Forest: {str(e)}")
        models['Random Forest'] = DummyModel(y.mean())
    
    # 4. Gradient Boosting Regression
    try:
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gbr.fit(X, y)
        models['Gradient Boosting'] = gbr
        logger.info("Trained Gradient Boosting model successfully")
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {str(e)}")
        models['Gradient Boosting'] = DummyModel(y.mean())
    
    # 5. K-Nearest Neighbors Regression
    try:
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X, y)
        models['KNN'] = knn
        logger.info("Trained KNN model successfully")
    except Exception as e:
        logger.error(f"Error training KNN: {str(e)}")
        models['KNN'] = DummyModel(y.mean())
    
    # Evaluate models
    for name, model in models.items():
        try:
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            logger.info(f"Model {name}: RMSE = {rmse:.2f}, R² = {r2:.2f}")
        except Exception as e:
            logger.error(f"Error evaluating {name}: {str(e)}")
    
    return models
```

## Configuration

### .streamlit/config.toml
**Purpose**: Configures Streamlit's behavior and appearance.

**Key Settings**:
- Server configuration (headless mode, port, address)
- Theme configuration (custom dark theme)
- Sidebar navigation settings

## Example Data Flow

### User Opens the Dashboard
1. `app.py` initializes the application
2. `data_processor.load_data()` loads and processes all data
3. `utils/theme.apply_theme()` applies the custom dark theme
4. `components/overview.show_overview()` displays the initial view

### User Clicks on Prediction Tool
1. `app.py` handles the navigation change
2. `components/prediction.show_prediction()` displays the prediction tool
3. User inputs project specifications
4. `models/predictor.predict_cpi()` makes a CPI prediction
5. `models/predictor.get_recommendation()` generates a recommendation
6. `models/predictor.simulate_win_probability()` calculates win probability
7. Results are displayed to the user

### User Clicks on Quick Insights
1. `utils/quick_insights.generate_quick_insights()` analyzes the data
2. Key metrics and trends are extracted
3. Insights are formatted with emojis and displayed