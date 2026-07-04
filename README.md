# Bitcoin Market Sentiment vs Hyperliquid Trader Performance Analysis

## Project Overview

This project analyzes the relationship between Bitcoin Fear & Greed market sentiment and Hyperliquid trader performance. Using historical trading data and daily market sentiment, it uncovers patterns in trader behavior, profitability, leverage usage, and risk-taking to generate actionable insights that can support smarter trading strategies.

## Objective

The objective is to explore how Bitcoin market sentiment influences trader performance by combining the Bitcoin Fear & Greed Index with Hyperliquid historical trading data. The project focuses on data preprocessing, exploratory data analysis (EDA), statistical analysis, and visualization to identify meaningful relationships between market sentiment and trading outcomes.

## Project Structure

```text
bitcoin-sentiment-trader-analysis/
│
├── data/
│   ├── fear_greed_index.csv
│   └── historical_data.csv
│
├── output/
│   ├── charts/
│   ├── merged_dataset.csv
│   ├── summary_statistics.csv
│   └── insights.txt
│
├── main.py
├── requirements.txt
├── README.md
└── report_template.md
```

## Technologies Used

* Python 3.12+
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Place both datasets inside the `data/` folder.
2. Execute the project:

```bash
python main.py
```

## Workflow

* Load and validate both datasets.
* Clean and preprocess the data.
* Engineer useful analytical features.
* Merge trading data with the Bitcoin Fear & Greed Index.
* Perform exploratory data analysis.
* Generate statistical summaries and visualizations.
* Produce business insights and recommendations.
* Export processed datasets and analysis results.

## Generated Outputs

* Merged dataset
* Summary statistics
* Professional visualizations
* Business insights report

All generated files are stored inside the `output/` directory.

## Future Improvements

* Predictive modeling using machine learning.
* Interactive dashboard using Streamlit or Dash.
* Real-time market sentiment integration.
* Automated trading performance monitoring.

## Author

Rohith Lakshman
