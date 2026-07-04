# Bitcoin Market Sentiment vs Hyperliquid Trader Performance Analysis

## Project Overview
This project analyzes the relationship between Bitcoin Fear & Greed market sentiment and Hyperliquid trader performance. The goal is to discover meaningful patterns, generate visualizations, perform statistical analysis, and provide actionable business insights that can improve trading strategies.

## Objective
To explore and quantify how varying states of market sentiment (Fear vs. Greed) impact trader profitability, leverage usage, and trade sizing on the Hyperliquid platform.

## Folder Structure
```
Primetrade_Assignment/
├── data/                       # Contains raw CSV datasets
│   ├── fear_greed_index.csv
│   └── historical_data.csv
├── output/                     # Contains generated datasets and insights
│   ├── charts/                 # Generated visualizations
│   ├── merged_dataset.csv
│   ├── summary_statistics.csv
│   └── insights.txt
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── report_template.md          # Template for final report
```

## Technologies Used
- **Python 3.12+**
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Computing)
- **Matplotlib & Seaborn** (Data Visualization)
- **SciPy** (Statistical Analysis)

## Installation

1. Clone or extract the project directory.
2. Ensure you have Python 3.12+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Place the datasets `fear_greed_index.csv` and `historical_data.csv` inside the `data/` folder.
2. Run the main script from the root directory:
   ```bash
   python main.py
   ```

## Project Workflow
1. **Data Loading**: Loads CSVs and handles parsing.
2. **Data Cleaning**: Removes duplicates, handles NAs, standardizes columns, fixes datetime formats.
3. **Feature Engineering**: Derives trade hour, weekday, absolute PnL, profit flags.
4. **Data Merging**: Joins trading data with daily sentiment values.
5. **EDA & Analysis**:
   - Overall Statistics
   - Sentiment Impact Analysis
   - Trader & Symbol Performance
   - Temporal (Time) Analysis
   - Buy vs Sell Comparison
   - Statistical testing (Correlation)
6. **Visualizations**: Auto-generates distribution plots, heatmaps, and bar charts.
7. **Business Insights**: Writes actionable insights to a text file.

## Generated Outputs
- **`summary_statistics.csv`**: Contains statistical descriptors for numeric columns.
- **`merged_dataset.csv`**: Cleaned and combined dataset.
- **`charts/`**: Directory containing all PNG visualizations.
- **`insights.txt`**: Automated business observations and recommendations.

## Key Findings
*(Please run `python main.py` and read `output/insights.txt` for the detailed generated findings).*

## Future Improvements
- Incorporate Machine Learning for PnL prediction based on sentiment.
- Add real-time API integrations for Live Order Book analysis.
- Build an interactive dashboard (e.g., using Streamlit or Dash) for dynamic filtering.
