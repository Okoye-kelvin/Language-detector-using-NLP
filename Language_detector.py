import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

NB = joblib.load('Language_detector.pkl')
CV = joblib.load('CountVectorizer.pkl')


st.write("<h1>Language Detection System</h1>", unsafe_allow_html=True)
st.write("Enter text to detect language")
user_input = st.text_area("Text Input", "Type your text here...", height = 200)


if st.button("Detect"):
    if user_input:
        user_input_clean = [user_input.lower()]
        user_input_transform = CV.transform(user_input_clean).toarray()

        prediction = NB.predict(user_input_transform)
        st.write(f"<h3>The detected language is: **{prediction[0]}**</h3>", unsafe_allow_html=True)
    else:
        st.write("Enter Text, to detect its language")
