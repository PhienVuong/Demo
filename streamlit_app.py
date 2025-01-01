import streamlit as st

st.title('Diamonds Price Prediction')

st.write('Upload your dataset here.')
file = st.file_uploader('Choose a CSV file', 'csv')
