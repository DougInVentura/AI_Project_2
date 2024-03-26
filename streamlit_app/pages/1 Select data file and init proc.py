# Not currently defined.
# I'm thinking this would
#       1) replace non null 'missing characters' such as "?" with pd.Nan, etc.
#       2) reduce label categories (use value counts to say how many categories and all others go into an 'other' which can get a custom name)
#               a) would help with all of the encoding and reduce dimentionality in the case of OHE.
#       3) convert any 'Object' columns that are in truth numeric to numeric (Int or Float)

import streamlit as st
import pandas as pd
import numpy as np
import os
import io



st.set_page_config(
    layout="wide",
    page_title="👋 Initial Processing Screen 👋",
    page_icon="👋"
)

st.write("""### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
### Page: **Select Data File and Init Proc**

Select csv file. It will upload to a dataframe and be saved initially with a RAW_ suffix in the data directory.
After selecting data file, go to page "2 Select Encoding Strategy"
  
""")

# initialize variables


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:  
    file_name = uploaded_file.name
    df_initial = pd.read_csv(uploaded_file, index_col=False)
    st.session_state['new_file'] = True


if 'new_file' in st.session_state and st.session_state['new_file'] == True:
    dir_and_raw_file_name = "data/" + "RAW_" + file_name
    df_initial.to_csv(dir_and_raw_file_name, index=False)
    dir_and_IN_PROCESS_file_name =  "data/" + "IN_PROCESS_" + file_name
    st.session_state['new_file'] = False
    st.session_state['df_in_process'] = df_initial
  

    # woulld have processing steps here to convert df_initial to df_ready_for_encoding.
    # for now, just copy it.
    df_ready_for_encoding = df_initial.copy()
    df_ready_for_encoding.to_csv(dir_and_IN_PROCESS_file_name, index=False)

    st.write("#### Top of Loaded dataframe")
    st.write(df_initial.head(10))
    
    # df.info() does not return anything to python, so need to pipe it
    buffer = io.StringIO() 
    df_initial.info(buf=buffer)
    s = buffer.getvalue()  
    st.text(f"#### {s}")

        
    st.text(f"dataframe value_counts \n{df_initial.value_counts()}")
   

