
import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import pdb
import plotly.express as px


st.set_page_config(
    layout="wide",
    page_title="👋 EDA and Graphs 👋",
    page_icon="👋"
)

if 'file_name' in st.session_state:
    file_name = st.session_state['file_name']

st.write(f"""### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
### Page: **EDA (Exploratory Data Analysis) and Graphs**

#### Working with file: :blue[{file_name}]
""")


if st.button("Branch to Preprocessor"):
    st.switch_page("pages/2 preprocessor.py")

# st.session_state['df_initial'] = df_initial
# st.session_state['df_initial_loaded'] = True
if 'df_initial_loaded' in st.session_state and st.session_state['df_initial_loaded'] and 'df_initial' in st.session_state:
    df_initial = st.session_state['df_initial']
    the_columns = df_initial.columns
    col1, col2 = st.columns([2,2])
    with col1:
        selected_y = st.selectbox(label = "#### **Select the Y column**", options = the_columns)
    with col2:
        selected_x = st.selectbox(label = "#### **Select the X column**", options = the_columns)
    st.write("## **_______________________________________________________________________________________________________**")
    if selected_y is not None and selected_x is not None:
        tab1, tab2 = st.tabs([f"##### **Scatter Chart {selected_y} vs {selected_x}**", "##### **Histogram of X**"])
        with tab1:
            st.write(f"#### **Scatter Plot of y={selected_y} vs x={selected_x}**")
            st.scatter_chart(data = df_initial, x= selected_x, y = selected_y)
        with tab2:
            fig = px.histogram(df_initial, x=selected_x)
            st.write(f"##### **Histogram of {selected_x}**")
            st.plotly_chart(fig, theme="streamlit")