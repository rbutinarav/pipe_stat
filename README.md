# Pipe Stat: Bootstrap Analysis for Sales Pipeline

This is a simple Streamlit app that allows you to perform bootstrap analysis on a dataset of opportunities. The app requires a CSV file with a specific structure, including the following columns:

- Opportunity Code
- Opportunity Description
- Customer ID
- Customer Description
- Category
- Value
- %
- Estimated Closing Date

The app will guide you through the process of loading the data, setting bootstrap parameters, running the bootstrap analysis, and displaying the results in both tabular and graphical formats.

## Features

- Load and display the dataset
- Modify bootstrap parameters
- Run bootstrap analysis
- Show dataset base statistics
- Display histogram of bootstrapped probability distribution of total value

## Installation

To run this app, you will need Python and the following libraries installed:

- streamlit
- pandas
- numpy
- matplotlib

You can install the required libraries using pip:

```bash
pip install streamlit pandas numpy matplotlib
```

## Usage

To run the app, use the following command in your terminal:

```bash
streamlit run app.py
```

Replace `app.py` with the name of the script containing the Streamlit app code if necessary.

## Contributing

Please feel free to submit issues or pull requests for improvements and bug fixes.