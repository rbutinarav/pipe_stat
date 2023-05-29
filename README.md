# Pipe Stat: Monte Carlo Analysis for Sales Pipelines

Pipe Stat is an interactive Streamlit application that performs Monte Carlo simulations on a sales pipeline dataset. This application requires a CSV file with a specific structure, including:

- Opportunity Code
- Opportunity Description
- Customer ID
- Customer Description
- Product Category
- Deal Value
- Deal Close Probability (%)
- Estimated Closing Date

The application guides users through the process of data import, Monte Carlo simulation parameter setting, simulation execution, and result interpretation in both tabular and graphical formats.

## Features

- Import and visualize your sales pipeline data
- Configure Monte Carlo simulation parameters
- Perform Monte Carlo simulation to predict total sales 
- Display basic statistical information about your data
- Graphically visualize the simulated probability distribution of total value

## Installation

To run this application, you need Python installed along with several Python libraries, including streamlit, pandas, numpy, and matplotlib.

You can install these required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the application, use the following command in your terminal:

```bash
streamlit run main.py
```

## Contributing

Contributions to improve the functionality and user experience of Pipe Stat are always welcome. If you're interested in contributing, feel free to submit issues or pull requests for improvements and bug fixes. Your input and expertise are greatly appreciated.
