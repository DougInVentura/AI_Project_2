import streamlit as st


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("""# 👋 Welcome to Pipey 👋
### Web-Based Application to Automate Supervised Machine Learning Classification Processing
#### With Preprocessing, Encoding, Imputation, Model Building and Scoring
#### Also includes Exploratory Data Analysis
### Project 2 ASU AI Course: Cody Cushing, David Gerhart, and Doug Francis
         
##### Version 1.0a """)


if st.button(":blue[Click Here] to select a csv file with your data to begin"):
    st.switch_page("pages/1 Select data file and init proc.py")


