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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import mlflow

#from stream_utils.widget import InputWidgetOption



def app():
    st.header("Title of the project")
    st.write("The projects aims to forcast ")
    dataset = list_datasets()
    #st.write(dataset)
    sel_col, selModel_col, upload_col = st.columns(3)

    dataset_selected = sel_col.selectbox("Datasets", options = dataset, index = 0)
    #st.write(dataset_selected)
    class_model = selModel_col.selectbox("Choose the model", options = get_models(), index = 0 )
    uploadede_file = upload_col.file_uploader("Upload your dataset", type=["csv"]) 


    #model_class = MODELS[class_model][model_class]
    m_class = MODELS[class_model]["model_class"]
    df = read_data(dataset_selected)

    #st.write(m_class)
    if class_model == "LinearRegression" :
       cont1 = st.container()
       col11, col12 = cont1.columns(2)
       fit_intercept = col11.selectbox("Choose fit_intercept", options = [True, False])
       normalize = col12.selectbox("Choose normalize", options = [True, False])
       col21, col22 = cont1.columns(2)
       positive = col21.selectbox("Choose positive", options = [True, False])
       
       regressor = LinearRegression(fit_intercept = fit_intercept, normalize = normalize, positive = positive)
       #model: BaseModel = LinearRegression()
       st.write(regressor)
       mlflow.autolog()
       if "model_name"  not in st.session_state:
         st.session_state["model_name"] = "LinearRegression"

       
     
    elif class_model == "RandomForestRegression" :
       cont1 = st.container()
       col11, col12 = cont1.columns(2)
       max_depth = col11.slider("Select max depth", min_value = 10, max_value= 100, value = 20, step = 10 )
       ccp_alpha = col12.slider("ccp_alpha", min_value = 0.01, max_value= 0.1, value = 0.02, step =  0.01 )

       col21, col22 = cont1.columns(2)
       n_estimator = col21.selectbox("Choose n_estimators", options = [50, 100, 150, 200, 250])
       random_state = col22.selectbox("Choose random_state", options= [50, 100, 150, 200, 250])

       
       col31, col32 = cont1.columns(2)
       criterion = col31.selectbox("Choose crietion", options = ["squared_error", "absolute_error", "poisson"])
       bootstrap = col32.selectbox("Choose bootstap", options = [True, False])
    
       col41, col42 = cont1.columns(2)
       n_jobs = col41.selectbox("Choose n_jobs", options = [10, 20, 30, 40, 50])
       regressor = RandomForestRegressor(max_depth = max_depth, ccp_alpha = ccp_alpha, n_estimators = n_estimator, random_state = random_state, criterion = criterion, bootstrap= bootstrap, n_jobs = n_jobs  )

       #model: BaseModel = RandomForestRegression()
       st.write(regressor)
       mlflow.autolog()
       if "model_name"  not in st.session_state:
         st.session_state["model_name"] = "RandomForestRegression"

  


    
    #st.write(model)
    #st.write(df)
    fs = st.multiselect(label = "select features", options = get_features(df))
    st.write(fs)
    if "features" not in st.session_state:
        st.session_state["features"] = fs

    size = st.slider("Select split size", min_value = 0.10, max_value= 0.90, value = 0.20, step = 0.10 )
    if st.button("PREDICT"):
        if len(fs) != 0:
            X = pd.DataFrame(df, columns = fs)
            X_train, X_test, y_train, y_test = train_test_split(X, df["price"], train_size = size)
            #st.write(X)
            regressor = regressor.fit(X_train, y_train)
            if "X_test" not in st.session_state:
                st.session_state["X_test"] = X_train

            #st.write(y_test)
            predictions = regressor.predict(X_test)
            st.subheader("Predictions")
            st.write(predictions)
            if "model"  not in st.session_state:
                st.session_state["model"] = regressor
            if "fs" not in st.session_state:
                st.session_state["fs"] = fs
        else:
            st.error("Select features")

   
   
    #train_test_values = model.splitData(X, df["price"], size)
    #st.write(train_test_values)
    #st.write(model)

    #widget_options =  model.get_options()

    """for key, value in widget_options.items():
         st.write(key)
         st.write(value)"""
    



    

        
    



