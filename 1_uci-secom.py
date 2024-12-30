import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from streamlit_gsheets import GSheetsConnection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import plotly.figure_factory as ff

## streamlit_setting
st.set_page_config(layout="wide",
                   page_title="UCI SECOM Dataset from Kaggle",
                   page_icon=":Chart:")

## connect data from google sheet
conn = st.connection("gsheets", type=GSheetsConnection)
secom = conn.read()



#Streamlit Header and Subheader
st.header('Data Analysis for the dataset - UCI-SECOM form Kaggle.')
st.text("""
Kaggle - UCI-SECOM Dataset
(https://www.kaggle.com/datasets/paresh2047/uci-semcom/data)

This page shows the data processing and each analysis steps.
The end goal is to predict the yield rate.
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
heatmap_corr = px.imshow(secom.iloc[:, 1:].corr())
st.plotly_chart(heatmap_corr, use_container_width=True)
secom_cleaned = secom.loc[:, (secom != 0).any(axis=0)]
print(secom_cleaned.shape)

## Drop Columns that have 70%+ correlation
st.subheader(":bulb: Drop Columns that have more than 70% correlation.")
st.text("""
After removing the columns that are all zeros, columns with high correlation will be removed.
(Only 196 columns left.)
""")

corr_matrix = secom_cleaned.iloc[:, 1:].corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find columns with correlation greater than 0.7
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.7)]
secom_dropped = secom_cleaned.drop(columns=to_drop)

## Present the cleaning dataset in the page using streamlit.df
st.text("""
And as the timestamp is not relevant to our data analysis(predict the yield)
The first column is finally removed in this step.
Below is the data after dropping columns:
""")
secom_dropped = secom_dropped.iloc[:, 1:]
print(secom_dropped.shape)
st.dataframe(secom_dropped)

## Split to Train and Test data
st.subheader(":bulb: Split Data into Train Dataset and Test Dataset")
st.text("""
80% to train; 20% to test.
(1253 rows in train and 314 rows to test)
The data has also been divided into X and Y. Y refer to the target column(Pass/Fail)
""")

df_x = secom_dropped.iloc[:, :194]
df_y = secom_dropped['Pass/Fail']
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42)
#To present to shape
dfs = {
    'x_train': x_train,
    'x_test': x_test,
    'y_train': y_train,
    'y_test': y_test
}
shapes_data = [(name, df.shape) for name, df in dfs.items()]
shapes_df = pd.DataFrame(shapes_data, columns=['DataFrame', 'Shape'])

st.text("""
Shapes of the Dataframes:
""")
st.table(shapes_df)

## XGBoost
st.subheader(":bulb: XGBoosts")
st.text("""
Data standardization has been made before training model.
""")

# Standardized
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Converting -1 to 0 in y_train and y_test
y_train = y_train.replace(-1, 0)
y_test = y_test.replace(-1, 0)

#  XGBoost
modelXG = XGBClassifier(random_state=1)
modelXG.fit(x_train, y_train)
y_predXG = modelXG.predict(x_test)

# Confusion Matrix
cmXG = confusion_matrix(y_test, y_predXG)
# Convert to a DataFrame for easier plotting
cmdf_XG = pd.DataFrame(cmXG, index=['Actual: Pass', 'Actual: Fail'],
                       columns=['Predicted: Pass', 'Predicted: Fail'])

# Create the confusion matrix heatmap
con_XGBoost = ff.create_annotated_heatmap(
    z=cmdf_XG.values,
    x=cmdf_XG.columns.tolist(),
    y=cmdf_XG.index.tolist(),
)

# Update layout for better readability
con_XGBoost.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted Labels',
    yaxis_title='Actual Labels'
)
# accuracy 92%
accuracy_xgboost = round(modelXG.score(x_test, y_test) * 100, 2)
st.text(f"""
Accuracy of the XGBoost Model: {accuracy_xgboost}%
""")
st.plotly_chart(con_XGBoost, use_container_width=True,
                key=1)


# RandomForest
st.subheader(":bulb: RandomForest")
st.text("""
Data standardization has been made before training model.
""")
# train model - Random Forest
modelRF = RandomForestClassifier(random_state=1)
modelRF.fit(x_train, y_train)
y_predRF = modelRF.predict(x_test)

# Confusion Matrix
cmRF = confusion_matrix(y_test, y_predRF)
# Convert to a DataFrame for easier plotting
cmdf_RF = pd.DataFrame(cmRF, index=['Actual: Pass', 'Actual: Fail'],
                       columns=['Predicted: Pass', 'Predicted: Fail'])

# Create the confusion matrix heatmap
con_RF = ff.create_annotated_heatmap(
    z=cmdf_RF.values,
    x=cmdf_RF.columns.tolist(),
    y=cmdf_RF.index.tolist(),
)

# Update layout for better readability
con_RF.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted Labels',
    yaxis_title='Actual Labels'
)
# accuracy 92%
accuracy_RF = round(modelRF.score(x_test, y_test) * 100, 2)

st.text(f"""
Accuracy of the Random Forest Model: {accuracy_RF}%
""")
st.plotly_chart(con_RF, key=2)


# Logistics Regression
st.subheader(":bulb: RandomForest")
st.text("""
Data standardization has been made before training model.
""")

# LogisticsRegression
LR = LogisticRegression(random_state=1)
LR.fit(x_train, y_train)
y_predLR = LR.predict(x_test)

# Confusion Matrix
cmLR = confusion_matrix(y_test, y_predLR)
# Convert to a DataFrame for easier plotting
cmdf_LR = pd.DataFrame(cmLR, index=['Actual: Pass', 'Actual: Fail'],
                       columns=['Predicted: Pass', 'Predicted: Fail'])

# Create the confusion matrix heatmap
con_LR = ff.create_annotated_heatmap(
    z=cmdf_LR.values,
    x=cmdf_LR.columns.tolist(),
    y=cmdf_LR.index.tolist(),
)

# Update layout for better readability
con_LR.update_layout(
    title='Confusion Matrix',
    xaxis_title='Predicted Labels',
    yaxis_title='Actual Labels'
)
# accuracy 87%
accuracy_LR = round(LR.score(x_test, y_test) * 100, 2)

st.text(f"""
Accuracy of the Random Forest Model: {accuracy_LR}%
""")
st.plotly_chart(con_LR, key=3)

# End
st.subheader("""
:Yawning Face: This is the end of the page.
Thank you for reading.
Please let me know if any advise.
📧vagary3316@gmail.com
""")

print("end")