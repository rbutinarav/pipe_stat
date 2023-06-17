import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Function to get risk emoji
def get_risk_emoji(risk):
    if risk > 80:
        return '🔴'  # return a red circle emoji for very high risk
    elif 50 < risk <= 80:
        return '🟠'  # return a orange circle emoji for high risk
    elif 20 < risk <= 50:
        return '⚠️'  # return a warning emoji for medium risk
    else:
        return '✅'  # return a green check emoji for low risk

# Function to initialize session state
def init_session_state():
    st.session_state.setdefault('dataset', None)
    st.session_state.setdefault('category_values', {})

# Function to load data
def load_data():
    dataset = st.file_uploader("Upload CSV", type=["csv"])
    if dataset is not None:
        dataset = pd.read_csv(dataset)
        dataset.columns = ['Opportunity Code','Opportunity Description','Customer ID','Customer Description','Product Category','Value','%','Estimated Closing Date']
        dataset['WeightedValue'] = dataset['Value'] * dataset['%'] / 100
        st.session_state['dataset'] = dataset
        st.session_state['category_values'] = {cat: [] for cat in dataset['Product Category'].unique()} 
    return dataset

# Function to display dataset
def display_dataset(dataset):
    st.subheader('Dataset')
    st.write(dataset)

# Function to run montecarlo
def run_montecarlo(iterations, dataset):
    category_values = {cat: [] for cat in dataset['Product Category'].unique()} 
    total_values = []
    
    progress_bar = st.progress(0)

    for i in range(iterations):
        category_totals = {cat: 0 for cat in dataset['Product Category'].unique()}

        for _, row in dataset.iterrows():
            value, probability = row['Value'], row['%'] / 100
            if random.random() < probability:
                category_totals[row['Product Category']] += value
                
        for category, total in category_totals.items():
            category_values[category].append(total)
        
        total_values.append(sum(category_totals.values()))
        progress_bar.progress((i + 1) / iterations)

    st.session_state['category_values'] = category_values
    st.session_state['total_values'] = total_values

# Function to display statistics
def display_statistics(dataset):
    total_values = st.session_state['total_values']
    category_values = st.session_state['category_values']
    st.subheader('Product Category Statistics') 

    # Initialize a list to hold the statistics for each category
    stats_list = []

    for category, values in category_values.items(): 
        if len(values) > 0:
            mean = np.mean(values) 
            p5 = np.percentile(values, 5) 
            p95 = np.percentile(values, 95)
            number_of_deals = len(dataset[dataset['Product Category'] == category])
            downside_risk = (mean - p5) / mean * 100 if mean > 0 else 0

            # Add a new dict to the list
            stats_list.append({"Category": category,
                            "Deals": number_of_deals,
                            "Mean": "{:,.2f}".format(mean),
                            "5th percentile": "{:,.2f}".format(p5),
                            "95th percentile": "{:,.2f}".format(p95),
                            "Downside risk (%)": "{:,.2f}".format(downside_risk),
                            #add a column with the risk emoji using get_risk_emoji(risk):
                            "Risk": get_risk_emoji(downside_risk)})
        else:
            # Handle case when there are no values
            stats_list.append({"Category": category,
                            "Mean": "N/A",
                            "5th percentile": "N/A",
                            "95th percentile": "N/A"})

    # Calculate statistics for the total pipeline value across all simulations
    mean = np.mean(total_values) 
    p5 = np.percentile(total_values, 5) 
    p95 = np.percentile(total_values, 95)
    number_of_deals = len(dataset)  # Total number of deals in the pipeline
    downside_risk = (mean - p5) / mean * 100 if mean > 0 else 0

    stats_list.append({"Category": "Total",
                    "Deals": number_of_deals,
                    "Mean": "{:,.2f}".format(mean),
                    "5th percentile": "{:,.2f}".format(p5),
                    "95th percentile": "{:,.2f}".format(p95),
                    "Downside risk (%)": "{:,.2f}".format(downside_risk),
                    "Risk": get_risk_emoji(downside_risk)})

    # Convert the list of dicts into a DataFrame
    stats_df = pd.DataFrame(stats_list)

    # Display the DataFrame
    st.table(stats_df)


# Function to show total distribution
def show_total_distribution(total_values):
    st.subheader('Total Distribution') 

    fig, ax = plt.subplots()

    weights = np.ones_like(total_values)/len(total_values)
    ax.hist(total_values, bins=100, weights=weights, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Projected Total Sales Value')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution of Projected Total Sales Value', fontsize=12)

    st.pyplot(fig)

# Function to show total distribution by category
def show_total_distribution_by_category(category_values):
    for category, values in category_values.items(): 
        if len(values) > 0:
            fig, ax = plt.subplots()

            weights = np.ones_like(values)/len(values)
            ax.hist(values, bins=100, weights=weights, color='blue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Projected Sales Value')
            ax.set_ylabel('Probability')
            ax.set_title(f'Distribution of Projected Sales Value for {category}', fontsize=12)

            st.pyplot(fig)

# Main program
def main():
    init_session_state()
    dataset = None

    st.write('Pipe Stat: This app will allow you to run a montecarlo analysis on a dataset of opportunities')

    if st.sidebar.checkbox('Load Data'):
        st.write('Dataset should have this struture: Opportunity Code,Opportunity Description,Customer ID,Customer Description, Product Category, Value, %, Estimated Closing Date')
        st.write('Headers will be reformatted in anycase, based on the structure above')
        dataset = load_data()

    if st.sidebar.checkbox('Show dataset'):
        display_dataset(st.session_state['dataset'])

    if st.sidebar.checkbox('Montecarlo Parameters'):
        iterations = st.sidebar.number_input('Number of iterations', min_value=1, max_value=100000, value=10000, step=1)
    else:
        iterations = 10000

    if st.sidebar.button('Run Montecarlo'):
        run_montecarlo(iterations, st.session_state['dataset'])

    if st.sidebar.checkbox('Show statistics'):
        display_statistics(st.session_state['dataset'])

    if st.sidebar.checkbox('Show Legend'):
        st.markdown('''
        ### Legend

        - **Category**: The product category that the deal falls into.
        - **Deals**: The total number of deals (or sales opportunities) currently open in this product category.
        - **Mean**: The mean value of the deals in this category.
        - **5th percentile**: The minimum value you can expect with a 95% confidence.
        - **95th percentile**: The maximum value you can expect with a 95% confidence.
        - **Downside risk (%)**: A measure of how much less the actual value could be compared to the mean value, with a 5% probability.
        - **Risk**: A visual representation of the downside risk.
        ''')

    if st.sidebar.checkbox('Show total distribution'): 
        show_total_distribution(st.session_state['total_values'])

    if st.sidebar.checkbox('Show total distribution by category'): 
        show_total_distribution_by_category(st.session_state['category_values'])

if __name__ == "__main__":
    main()
