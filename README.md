# CPI Analysis & Prediction Dashboard

A sophisticated Streamlit dashboard for predictive pricing analysis in market research, featuring advanced data quality tools and interactive visualizations.

## Features

- **Interactive Analytics**: Explore CPI trends across different market segments with dynamic visualizations
- **Predictive Pricing**: Advanced machine learning models including Ridge, ElasticNet, Random Forest, and Gradient Boosting to predict optimal CPI based on project specifications
- **Strategic Insights**: Data-driven recommendations for pricing strategy optimization
- **Data Quality Analysis**: Comprehensive tools for data validation and outlier detection
- **Advanced Model Metrics**: Learning curves, residual analysis, and feature importance for model explainability
- **Model Monitoring**: Track prediction accuracy and detect model drift over time
- **Quick Insights**: Generate emoji-powered summaries of key data trends instantly

## Dashboard Sections

- **Overview**: High-level summary of key metrics and CPI comparison between won and lost bids
- **Detailed Analysis**: In-depth analysis of how IR, LOI, and sample size affect CPI
- **CPI Prediction**: Interactive tool for predicting optimal CPI for new bids
  - **Advanced Model Metrics**: Learning curves, residuals analysis, and feature dependence plots to explain model behavior
  - **Model Diagnostics**: Tools to detect overfitting/underfitting and evaluate model reliability
  - **Feature Importance Analysis**: Visual explanation of which factors most influence CPI predictions
- **Insights & Recommendations**: Strategic recommendations based on data analysis
- **Model Monitoring**: Track model performance metrics over time

## Technology Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: 
  - Scikit-learn (Ridge, ElasticNet, Random Forest, Gradient Boosting)
  - Advanced feature engineering (polynomial features, interaction terms)
  - Hyperparameter optimization
- **Data Sources**: Excel processing with OpenpyXL

## Getting Started

See the [Installation & Setup](documentation.md#installation--setup) section in the documentation for detailed instructions.

## Documentation

For comprehensive documentation on the codebase, architecture, and features, see [documentation.md](documentation.md).

## Screenshots

![Dashboard Overview](screenshots/dashboard_overview.png)
*Main dashboard overview showing key metrics*

![CPI Prediction Tool](screenshots/prediction_tool.png)
*Interactive prediction tool for estimating optimal CPI*

## License

[MIT License](LICENSE)