import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

#streamlit_setting

st.set_page_config(layout="wide",
                   page_title="UCI-SECOM Dataset from Kaggle",
                   page_icon=":Chart:")

#streamlit title&Time Frame
st.header('Data Analysis for the dataset - UCI-SECOM form Kaggle.')
st.caption("""
Hi, this is my data visualization project.
The data comes from Kaggle - UCI-SECOM Dataset
link: https://www.kaggle.com/datasets/paresh2047/uci-semcom/code?datasetId=28901&sortBy=commentCount
""")