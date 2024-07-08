import streamlit as st
from Trainer import Trainer
from Scorer import Scorer
import os

# To run!
# streamlit run "c:/Users/Manuel/OneDrive - ITBA/ITBA/Año 3/2º Cuatrimestre/Análisis Predictivo/Final/streamlit_main.py"
st.title('Model Management Application')

option = st.selectbox(
    'Choose an option:',
    ['Train the model', 'Score the model', 'Train and Score the model']
)

if st.button('Proceed'):
    if option == 'Train the model':
        trainer = Trainer('Data/online_shoppers_intention.csv')
        trainer.orchestrator()
        st.success('Model trained successfully!')
    elif option == 'Score the model':
        scorer = Scorer('Data/test.csv')
        filename=scorer.orchestrator()
        st.success('Model scored successfully!')
        with open(filename, "rb") as file:
            btn = st.download_button(
                label="Download CSV",
                data=file,
                file_name="scored_data.csv",
                mime="text/csv",
            )   

    elif option == 'Train and Score the model':
        trainer = Trainer('Data/online_shoppers_intention.csv')
        trainer.orchestrator()
        scorer = Scorer('Data/test.csv')
        filename=scorer.orchestrator()
        st.success('Model trained and scored successfully!')
        with open(filename, "rb") as file:
            btn = st.download_button(
                label="Download CSV",
                data=file,
                file_name="scored_data.csv",
                mime="text/csv",
            )   




st.write('For MLFlow UI:')
st.write('1. Paste this code into a terminal: `mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db`')
st.write('2. Follow this link: `http://127.0.0.1:8080`')
st.write('Thank you for using the application!')