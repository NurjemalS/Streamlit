import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def app():
    st.header("Reports")
    