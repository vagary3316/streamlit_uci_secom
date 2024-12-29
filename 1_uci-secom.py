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
1567  semi-conductor manufacturing data examples with 591 features.
The first column is timestamp; last column is the target column with values either -1 or 1.
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

secom = secom.replace(np.nan, 0)

## Present the original Pass rate of the secom dataset
st.subheader(":bulb: Pie Chart of the Pass/Fails ")
st.text("""
The last columns ['Pass/Fail'] is the target column. 
-1 means Pass; 1 means Fail
There are 1,463 Pass, 104 Fails. (93.4% Pass)
""")
yield_df = pd.DataFrame(secom['Pass/Fail'].value_counts()).reset_index()
pie_for_yield = px.pie(yield_df, values='count', names='Pass/Fail', color_discrete_map='Pastel')
pie_for_yield.update_traces(textposition='inside', textinfo='percent+label')
pie_for_yield.update_layout(showlegend=True)  # Enable the legend
st.plotly_chart(pie_for_yield, use_container_width=True)

## Correlation Heatmap
st.subheader(":bulb: Correlation Heatmap for the features")
st.text("""
As there are a few features in the dataset, we should investigate collinearity using heatmap of correlation.
And the heatmap reminds that there are some columns that only contains zero in their values.
(We replaced Na with zero before.)
After removing the columns that are all zeros, there are 480 columns left.(Timestamp and Pass/Fail included)
""")
heatmap_corr=px.imshow(secom.iloc[:, 1:].corr())
st.plotly_chart(heatmap_corr, use_container_width=True)
secom_cleaned = secom.loc[:, (secom != 0).any(axis=0)]
print(secom_cleaned.shape)

## Drop Columns that have 70%+ correlation
st.subheader(":bulb: Drop Columns that have more than 70% correlation.")
st.text("""
After removing the columns that are all zeros, columns with high correlation will be removed.
Only 196 columns left.)
""")

corr_matrix = secom_cleaned.iloc[:,1:].corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find columns with correlation greater than 0.7
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.7)]
secom_dropped = secom_cleaned.drop(columns=to_drop)
print(secom_dropped.shape)
