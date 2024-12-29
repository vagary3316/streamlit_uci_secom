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
st.text("""
Kaggle - UCI-SECOM Dataset
(https://www.kaggle.com/datasets/paresh2047/uci-semcom/data)
""")

st.subheader(":bulb: Data Description:")
st.text("""
1567  semi-conductor manufacturing data examples with 591 features
The last columns ['Pass/Fail'] is the target column: -1 means Pass; 1 means Fail
And total 41951 NA in the dataset.
""")

## Present the whole dataset in the page using streamlit.df
st.dataframe(secom)

## Explain NAs, and will replace NA with 0
st.subheader(":bulb: Deal with NA:")
st.text("""
Replace with 0
The values are not presented, may mean that there is no signal detected for the sensor.
""")

secom.replace(np.nan, 0)

## Present the original Pass rate of the secom dataset
st.subheader(":bulb: Pie Chart of the Pass/Fails ")
st.text("""
There are 1,463 Pass, 104 Fails.
93.4% Pass
""")
yield_df = pd.DataFrame(secom['Pass/Fail'].value_counts())
yield_df.rename(index={-1: 'Pass', 1: 'Fail'}, inplace=True)
pie_for_yield = px.pie(yield_df, values='count',
                       color_discrete_map='Pastel',
                       labels = yield_df.index
)
pie_for_yield.update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(pie_for_yield, use_container_width=True)

