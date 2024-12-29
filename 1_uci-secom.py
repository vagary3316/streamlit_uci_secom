import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_gsheets import GSheetsConnection


## connect data from google sheet
conn = st.connection("gsheets", type=GSheetsConnection)
secom = conn.read()


## streamlit_setting
st.set_page_config(layout="wide",
                   page_title="UCI SECOM Dataset from Kaggle",
                   page_icon=":Chart:")

#Streamlit Header and Subheader
st.header('Data Analysis for the dataset - UCI-SECOM form Kaggle.')
st.markdown("""
Kaggle - UCI-SECOM Dataset
(https://www.kaggle.com/datasets/paresh2047/uci-semcom/code?datasetId=28901&sortBy=commentCount)
""")

st.markdown("""
:bulb: Data Description:
1567  semi-conductor manufacturing data examples with 591 features
The last columns ['Pass/Fail'] is the target column: -1 means Pass; 1 means Fail
And total 41951 NA in the dataset.
""")

## Present the whole dataset in the page using streamlit.df
st.dataframe(secom)

## Explain NAs, and will replace NA with 0
st.markdown("""
:bulb: Deal with NA:
Replace with 0
The values are not present, may mean that there is no signal detected for the sensor.
""")

secom.replace(np.nan, 0)

## Present the original Pass rate of the secom dataset
st.markdown("""
:bulb: Pie Chart of the Pass/Fails 

""")

yield_df = secom['Pass/Fail'].value_counts()
pie_for_yield = px.pie(yield_df, values='count', title='Pass/Fail Distribution', color_discrete_map='Pastel')
pie_for_yield.update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(pie_for_yield, use_container_width=True)

