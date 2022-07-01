from queue import Empty
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from stream_utils.utils import list_datasets, read_data, get_features
from configuration import MODELS, get_models
from models.base import BaseModel
from models.RandomForestRegression import RandomForestRegression
from models.LinearRegression import LinearRegression
#from stream_utils.widget import InputWidgetOption



def app():
    st.header("Title of the project")
    st.sidebar.slider("Select max depth", min_value = 10, max_value= 100, value = 20, step = 10 )
    st.write("The projects aims to forcast ")
    dataset = list_datasets()
    st.write(dataset)
    sel_col, selModel_col, upload_col = st.columns(3)

    dataset_selected = sel_col.selectbox("Datasets", options = dataset, index = 0)
    st.write(dataset_selected)
    class_model = selModel_col.selectbox("Choose the model", options = get_models(), index = 0 )
    upload_col.file_uploader("Upload your dataset", type=["csv"]) 

    #model_class = MODELS[class_model][model_class]
    m_class = MODELS[class_model]["model_class"]
    df = read_data(dataset_selected)
    st.write(m_class)
    model: BaseModel = RandomForestRegression()
    
    st.write(model)
    st.write(df)
    fs = st.multiselect(label = "select features", options = get_features(df))
    st.write(fs)

    
    
    if len(fs) != 0:
        X = pd.DataFrame(df, columns = fs)
        st.write(X)
    
    size = st.slider("Select split size", min_value = 0.10, max_value= 0.90, value = 0.20, step = 0.10 )
   
    X_train, X_test, y_train, y_test = train_test_split(X, df["price"], train_size = 0.20)
    #train_test_values = model.splitData(X, df["price"], size)
    #st.write(train_test_values)
    #st.write(model)

    model = model.fit(X_train, y_train)
    st.write(y_test)
    predictions = model.predict(X_test)
    st.write(predictions)
   




    #widget_options =  model.get_options()

    """for key, value in widget_options.items():
         st.write(key)
         st.write(value)"""
    



    

        
    



