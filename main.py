import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def get_risk_emoji(risk):
    if risk > 80:
        return '🔴'  # return a red circle emoji for very high risk
    elif 50 < risk <= 80:
        return '🟠'  # return a orange circle emoji for high risk
    elif 20 < risk <= 50:
        return '⚠️'  # return a warning emoji for medium risk
    else:
        return '✅'  # return a green check emoji for low risk
    

#inizialized the session state
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'projected_value' not in st.session_state:
    st.session_state['projected_value'] = None
if 'category_values' not in st.session_state:
    st.session_state['category_values'] = {}

dataset = st.session_state['dataset']
projected_value = st.session_state['projected_value']
category_values = st.session_state['category_values']

st.write ('Pipe Stat: This app will allow you to run a montecarlo analysis on a dataset of opportunities')
#ask user to load the data
load_data = st.sidebar.checkbox('Load Data')

if load_data:
    dataset = st.file_uploader("Upload CSV", type=["csv"])
    #convert the above loaded file into a pandas dataframe
    
    st.write('Dataset should have this struture: Opportunity Code,Opportunity Description,Customer ID,Customer Description, Product Category, Value, %, Estimated Closing Date')
    st.write('Headers will be reformatted in anycase, based on the structure above')

    if dataset is not None:
        dataset = pd.read_csv(dataset)
        #change the name of the columns based on their position as follows: Opportunity Code,Opportunity Description,Customer ID,Customer Description,Product Category,Value,%,Estimated Closing Date
        dataset.columns = ['Opportunity Code','Opportunity Description','Customer ID','Customer Description','Product Category','Value','%','Estimated Closing Date']
        #calculate the weighted value
        dataset['WeightedValue'] = dataset['Value'] * dataset['%'] / 100
        st.session_state['dataset'] = dataset
        st.session_state['category_values'] = {cat: [] for cat in dataset['Product Category'].unique()} 


#ask the user if want to see the dataset (check box)
if st.sidebar.checkbox('Show dataset'):
    st.subheader('Dataset')
    dataset = st.session_state['dataset']
    st.write(dataset)


#ask the user to modify the montecarlo parameters
montecarlo_param = st.sidebar.checkbox('Montecarlo Parameters')
if montecarlo_param:
    category_values = st.session_state['category_values']  # Fetch updated category_values from session state
    total_values = []
    dataset = st.session_state['dataset']
    #ask users to define the number of iterations
    iterations = st.number_input('Number of iterations', min_value=1, max_value=100000, value=10000, step=1)
    sample_size = len(dataset)

#ask ths user if wants to run the Montacarlo
run_montecarlo = st.sidebar.button('Run Montecarlo')


if run_montecarlo:
    # Initialize a dictionary to hold the projected values for each category
    st.session_state['category_values'] = {cat: [] for cat in dataset['Product Category'].unique()} 
    st.session_state['total_values'] = []
    # Initialize a dictionary to hold the projected values for each category
    category_values = {cat: [] for cat in dataset['Product Category'].unique()} 
    total_values = []
    
    progress_bar = st.progress(0)

    for i in range(iterations):
        # Initialize a dictionary to hold the total value for each category within the current simulation
        category_totals = {cat: 0 for cat in dataset['Product Category'].unique()}

        for _, row in dataset.iterrows():
            value, probability = row['Value'], row['%'] / 100
            if random.random() < probability:
                category_totals[row['Product Category']] += value
                
        for category, total in category_totals.items():
            category_values[category].append(total)
        
        total_values.append(sum(category_totals.values()))
        st.session_state['projected_value'] = total_values
        st.session_state['category_values'] = category_values  # Store updated category_values back to session state
        progress_bar.progress((i + 1) / iterations)
        st.session_state['total_values'] = total_values
    

if st.sidebar.checkbox('Show statistics'):
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
            downside_risk = (mean - p5) / mean * 100

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


if st.sidebar.checkbox('Show total distribution'): 
    st.subheader('Total Distribution') 

    fig, ax = plt.subplots()

    weights = np.ones_like(total_values)/len(total_values)
    ax.hist(total_values, bins=100, weights=weights, color='blue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Projected Total Sales Value')
    ax.set_ylabel('Probability')
    ax.set_title('Distribution of Projected Total Sales Value', fontsize=12)

    st.pyplot(fig)


if st.sidebar.checkbox('Show total distribution by category'): 

    for category, values in category_values.items(): 
        if len(values) > 0:
            fig, ax = plt.subplots()

            weights = np.ones_like(values)/len(values)
            ax.hist(values, bins=100, weights=weights, color='blue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Projected Sales Value')
            ax.set_ylabel('Probability')
            ax.set_title(f'Distribution of Projected Sales Value for {category}', fontsize=12)

            st.pyplot(fig)

