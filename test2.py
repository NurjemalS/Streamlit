import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Using object notation
add_selectbox = st.sidebar.selectbox(
    "Page Navigation",
    ("Main","Training and Prediction", "Reports", "Simulation and Integration")
)

def training_predict():
    import streamlit as st
    st.write("Title")




