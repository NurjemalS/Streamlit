
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()
explain = st.button("Explaination")

@st.cache
def get_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


with header:
    st.title("ProjectTitle")
    st.text("Explanation purpose of the project")

with dataset:
    st.header("this is the dataset")
    st.text("dataset explanation")
    dataset = get_data("data.csv")
    st.write(dataset.head())

with features:
    st.header("Features")
    st.text("Description")



with model_training:
    st.header("train the model")
    st.text("Description")

    sel_col, disp_col = st.columns(2)

    max_depth = sel_col.slider("Select max depth", min_value = 10, max_value= 100, value = 20, step = 10 )
    n_estimators = sel_col.selectbox("How many trees should be there", options = [50, 100, 150, 200, 250, 'No limit'], index = 0)
    sel_col.write("these are features")
    price_data = dataset["price"]
    dataset = dataset.drop(["date", "price", "street", "city", "statezip", "country"], axis = 1)
    sel_col.write(dataset.columns)


    input_feature = sel_col.text_input("which features should be used as the input", 'bedrooms')
    #sel_col.write(input_feature)
    
    features = input_feature.split(",")
    sel_col.write(features)
    if n_estimators == 'No limit':
        reg = RandomForestRegressor(max_depth = max_depth)
    else:
         reg = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)
    
    
    X = dataset[features]
    y = price_data

    reg.fit(X,y)
    prediction = reg.predict(X)

    disp_col.subheader("Mean absolute error")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("Mean squared error")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("R  squared score")
    disp_col.write(r2_score(y, prediction))


if explain:
    
    explainer =  shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)

