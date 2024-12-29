import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

#streamlit_setting

st.set_page_config(layout="wide",
                   page_title="UCI-SECOM Dataset from Kaggle",
                   page_icon=":Chart:")

#streamlit title&Time Frame
st.title('Data Analysis for the dataset - UCI-SECOM form Kaggle.')