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

st.set_page_config(layout="wide")

st.write("""# 👋 Initial Processing Screen 👋
### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
         

Select csv file. It will upload to a dataframe and be saved initially with a RAW_ suffix in the data directory.
After Initial preprocessing, the file created will be data\INIT_PROCESS_file_name.csv
INIT_PROCESS_file_name.csv will be used for "2 Select Encoding Strategy"
  
""")

# initialize variables
st.session_state['new_fie'] = False


uploaded_file = st.file_uploader("Choose a file")
# might want as a diagnosic shown only w button click: st.write(uploaded_file)
if uploaded_file is not None:  
    file_name = uploaded_file.name
    df_initial = pd.read_csv(uploaded_file, index_col=False)
    st.dataframe(df_initial.head(7))
    st.session_state['new_fie'] = True

if st.session_state['new_fie'] == True:
    dir_and_raw_file_name = "data/" + "RAW_" + file_name
    df_initial.to_csv(dir_and_raw_file_name, index=False)
    dir_and_INIT_PROC_file_name =  "data/" + "INIT_PROC_" + file_name

    # woulld have processing steps here to convert df_initial to df_ready_for_encoding.
    # for now, just copy it.
    df_ready_for_encoding = df_initial.copy()
    df_ready_for_encoding.to_csv(dir_and_INIT_PROC_file_name, index=False)

    st.write("#### Top of dataframe")
    st.write(df_initial.head(10))
    
    # df.info() does not return anything to python, so need to pipe it
    
    
    buffer = io.StringIO() 
    df_initial.info(buf=buffer)
    s = buffer.getvalue()  
    st.text(f"#### {s}")

    
    
    the_char= st.text_input("Enter the missing char code (will substitute NULLs in its place)")
    if len(the_char) > 1:
        st.text(f"Needed single character received {the_char}, which is length of {len(the_char)}")
        ready_2_replace = False
    elif len(the_char) <= 0:
        ready_2_replace = False
        st.text(f"Did not receive a character. Enter a character to replace")
    else:
        ready_2_replace = True
        st.text("would be ready to replace")
        
    
   

