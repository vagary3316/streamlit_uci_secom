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

#streamlit title
st.header('Data Analysis for the dataset - UCI-SECOM form Kaggle.')
st.text("""
Hi, this is my data visualization project.\n
The data comes from Kaggle - UCI-SECOM Dataset
(https://www.kaggle.com/datasets/paresh2047/uci-semcom/code?datasetId=28901&sortBy=commentCount)
""")


#test
st.text(df.shape)