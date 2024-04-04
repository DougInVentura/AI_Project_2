
import streamlit as st
from streamlit_extras.no_default_selectbox import selectbox as selectbox_no_default
import pandas as pd
import numpy as np
import os
import io
import pdb
import plotly.express as px


pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

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
    col1, col2, col3, col4 = st.columns([2,2,2,2])
    with col1:
        selected_x = st.selectbox(label = "#### **Select the X column**", options = the_columns)
    with col2:
        selected_y = st.selectbox(label = "#### **Select the Y column**", options = the_columns)
    with col3:
        selected_color = selectbox_no_default(label = "#### **Scatter plot Color by**", options = the_columns)
    with col4:
        selected_size = selectbox_no_default(label = "#### **Scatter Plot Size by**", options = the_columns)
    st.write("## **_______________________________________________________________________________________________________**")
    if selected_x is not None:
        tab1, tab2, tab3 = st.tabs([f"##### **Scatter Chart {selected_y} vs {selected_x}**", "##### **Histogram of X**","##### Correlation Matrix for Numeric"])
        with tab1:
            if selected_y is not None:
                if selected_color is not None:
                    if selected_size is not None:
                        st.write(f"#### **Scatter Plot of y={selected_y} vs x={selected_x}, color by={selected_color}, size by={selected_size}**")
                        st.scatter_chart(data = df_initial, x= selected_x, y = selected_y, color=selected_color, size=selected_size)
                    else: 
                        st.write(f"#### **Scatter Plot of y={selected_y} vs x={selected_x}, color by={selected_color}**")
                        st.scatter_chart(data = df_initial, x= selected_x, y = selected_y, color=selected_color)
                else:
                    st.write(f"#### **Scatter Plot of y={selected_y} vs x={selected_x}**")
                    st.scatter_chart(data = df_initial, x= selected_x, y = selected_y)
        with tab2:
            fig = px.histogram(df_initial, x=selected_x)
            st.write(f"##### **Histogram of {selected_x}**")
            st.plotly_chart(fig, theme="streamlit")
        with tab3:
            st.write(f"##### **Correlation Matrix for Numeric Columns**")
            st.text(df_initial.select_dtypes(include='number').corr(min_periods=12))
    else:
        st.write("Select columns, starting with X")
                