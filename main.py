import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#ask user to load a dataset with the following structure:
#Opportunity Code,Opportunity Description,Customer ID,Customer Description, Category, %, Estimated Closing Date
#format is csv

#inizialized the session state
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None
if 'projected_value' not in st.session_state:
    st.session_state['projected_value'] = None

dataset = st.session_state['dataset']
projected_value = st.session_state['projected_value']

st.write ('Pipe Stat: This app will allow you to run a montecarlo analysis on a dataset of opportunities')
#ask user to load the data
load_data = st.sidebar.checkbox('Load Data')

if load_data:
    dataset = st.file_uploader("Upload CSV", type=["csv"])
    #convert the above loaded file into a pandas dataframe
    
    st.write('Dataset should have this struture: Opportunity Code,Opportunity Description,Customer ID,Customer Description, Category, Value, %, Estimated Closing Date')
    st.write('Headers will be reformatted in anycase, based on the structure above')

    if dataset is not None:
        dataset = pd.read_csv(dataset)
        #change the name of the columns based on their position as follows: Opportunity Code,Opportunity Description,Customer ID,Customer Description, Category, %, Estimated Closing Date
        dataset.columns = ['Opportunity Code','Opportunity Description','Customer ID','Customer Description', 'Category', 'Value', '%', 'Estimated Closing Date']
        #calculate the weighted value
        dataset['WeightedValue'] = dataset['Value'] * dataset['%'] / 100
        st.session_state['dataset'] = dataset
        #change the name of the columns


#ask the user if want to see the dataset (check box)
if st.sidebar.checkbox('Show dataset'):
    st.subheader('Dataset')
    dataset = st.session_state['dataset']
    st.write(dataset)


#ask the user to modify the montecarlo parameters
montecarlo_param = st.sidebar.checkbox('Montecarlo Parameters')
if montecarlo_param:
    # Define the number of iterations and the sample size
    #ask users to define the number of iterations
    iterations = st.number_input('Number of iterations', min_value=1, max_value=100000, value=10000, step=1)
    sample_size = len(dataset)

#ask ths user if wants to run the Montacarlo
run_montecarlo = st.sidebar.button('Run Montecarlo')


if run_montecarlo:
    total_values = []

    dataset = st.session_state['dataset']
    
    # Create a progress bar
    progress_bar = st.progress(0)

    for i in range(iterations):
        total_value = 0
        
        for _, row in dataset.iterrows():
            value, probability = row['Value'], row['%'] / 100
            
            if random.random() < probability:
                total_value += value
                
        total_values.append(total_value)
        st.session_state['projected_value'] = total_values

        # Update the progress bar
        progress_bar.progress((i + 1) / iterations)



#ask user to show dataset base statistics
if st.sidebar.checkbox('Show dataset base statistics'):
    st.subheader('Dataset base statistics')
    #st.write(dataset.describe())


    # Calculate the statistics (e.g., mean, median, percentiles)
    mean = np.mean(projected_value)
    median = np.median(projected_value)
    p5 = np.percentile(projected_value, 5)
    p95 = np.percentile(projected_value, 95)
    downside_risk = (mean - p5) / mean

    #format in currency format
    mean = "{:,.2f}".format(mean)
    median = "{:,.2f}".format(median)
    p5 = "{:,.2f}".format(p5)
    p95 = "{:,.2f}".format(p95)
    #formate downside risk in percentage
    downside_risk = downside_risk * 100
    downside_risk = "{:,.2f}".format(downside_risk)

    #display basic statistics
    #number of deals
    st.write(f'Number of deals: {len(dataset)}')
    #total nominal value of sales
    st.write(f'Total nominal value of sales: {dataset["Value"].sum()}')
    #totale weighted value of sales
    st.write(f'Total weighted value of sales: {dataset["WeightedValue"].sum()}')
    #confidence range of the weighted value of sales
    st.write(f'Confidence range of the weighted value of sales: {p5} - {p95}')

    st.write('-- additional statistics --')

    # Display the statistics using Streamlit
    st.write(f"Mean: {mean}")
    st.write(f"Median: {median}")
    st.write(f"5th percentile: {p5}")
    st.write(f"95th percentile: {p95}")

    # Plot the histogram of the total values
    fig, ax = plt.subplots()
    ax.hist(projected_value, bins=100, density=True, color='green', edgecolor='grey')
    ax.set_xlabel('Projected Sales Value')
    ax.set_ylabel('Probability Density')
    #ax.set_title('montecarloped Probability Distribution of Total Value')
    #set a title with littler font
    ax.set_title('Distribution of Projected Sales Value', fontsize=10)

    st.write(f'This means that there is a 95% chance that sales value of the pipeline will be more than {p5} and less than {p95}')

    #calculate the difference between the mean and p5 in percentage and call it downside risk
    
    st.write(f'This means that there is a 5% chance that sales value of the pipeline will be less than {downside_risk}% of the mean value of {mean}')

    st.pyplot(fig)


    