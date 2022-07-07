import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




def app():
    st.header("Reports")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    if "model"  not in st.session_state:
           st.write("first you need to predict")
           

    else:
        model = st.session_state["model"]
     

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(st.session_state["X_test"])
        st.header("Feature Importance")
        plt.title("Feature importance based on shap values")
        shap.summary_plot(shap_values, st.session_state["X_test"])
        st.pyplot(bbox_inches='tight', dpi=300,pad_inches=0)
        plt.clf()
        st.write("---")

        plt.title("Feature importance based on shap values (Bar)")
        shap.summary_plot(shap_values, st.session_state["X_test"], plot_type="bar")
        st.pyplot(bbox_inches='tight', dpi=300,pad_inches=0)
        plt.clf()
        st.write("---")
        st.write("---")

        feature = st.selectbox("Select feature dor Dependence_plot", options=st.session_state["fs"])
        shap.dependence_plot(feature, shap_values,st.session_state["X_test"])
        st.pyplot(bbox_inches='tight', dpi=300,pad_inches=0)
        plt.clf()



        """st.write(st.session_state["model"])
        st.write(st.session_state["model_name"])
        regressor = st.session_state["model"]
        f_importance = pd.DataFrame(regressor.feature_importances_, index = st.session_state["fs"] ,columns=['importance_value']).sort_values(by='importance_value',ascending=False)
        st.write(f_importance)
        fig = f_importance.plot(kind = 'bar')
        #st.pyplot(fig)
        #plt.show()      
        st.bar_chart(f_importance, width=100, height=350)
        cont = st.container()"""






      

        
    