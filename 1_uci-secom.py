import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_gsheets import GSheetsConnection


## connect data from google sheet
conn = st.connection("gsheets", type=GSheetsConnection)
df = conn.read()


## streamlit_setting
st.set_page_config(layout="wide",
                   page_title="UCI-SECOM Dataset from Kaggle",
                   page_icon=":Chart:")

#Streamlit Header and Subheader
st.header('Data Analysis for the dataset - UCI-SECOM form Kaggle.')
st.text("""
Kaggle - UCI-SECOM Dataset
(https://www.kaggle.com/datasets/paresh2047/uci-semcom/code?datasetId=28901&sortBy=commentCount)
""")

st.subheader("""
Data Description:
1567  semi-conductor manufacturing data examples with 591 features
The last columns ['Pass/Fail'] is the target column: -1 means Pass; 1 means Fail
""")
