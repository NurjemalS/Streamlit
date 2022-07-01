import streamlit as st
from multiapp import MultiApp
from apps import train_predict, reports, simulation_iteration
# Initial page config

st.set_page_config(
    page_title="Forecasting Experimentation Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

app = MultiApp()

# Add all your application here
app.add_app("Training and Prediction", train_predict.app)
app.add_app("Reports", reports.app)
app.add_app("Simulation and Iteration", simulation_iteration.app)

# The main app
app.run()