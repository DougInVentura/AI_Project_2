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
import pdb



st.set_page_config(
    layout="wide",
    page_title="👋 Initial Processing Screen 👋",
    page_icon="👋"
)

st.write("""### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
### Page: **Select Data File and Init Proc**

Select csv file. It will upload to a dataframe, df_initial, and be saved archived with a RAW_ suffix in the data directory.
After selecting data file, use the drop down to select and separate out the 'y' variable. Then goto "2 Select Encoding Strategy"
""")

# initialize variables
if 'file_name' in st.session_state:
    file_name = st.session_state['file_name']

container_1 = st.empty()
with container_1:
    uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:  
    file_name = uploaded_file.name
    st.session_state['file_name'] = file_name
    df_initial = pd.read_csv(uploaded_file, index_col=False)
    st.session_state['new_file'] = True
    st.session_state['df_initial'] = df_initial
    st.session_state['df_initial_loaded'] = True
    st.write(f"#### Working with file :blue[{file_name}]")
    # below we split into X and Y. we will not initialize df_loaded until that is done and in the session state.


if 'new_file' in st.session_state and st.session_state['new_file'] == True:
    # start with identifying the y variable and separating it. 
    col_list = df_initial.columns.to_list()
    y_select_container = st.empty()
    option = y_select_container.selectbox('Select Y variable to separate out',col_list, index = None)
    if option is not None:
        # write out entire dataframe w RAW_ suffix
        dir_and_raw_file_name = "data/" + "RAW_" + file_name
        df_initial.to_csv(dir_and_raw_file_name, index=False)
        y_column = option
        # write y out below

        # set up X in process and y in process dataframes and save them
        # includes 1) create the X and Y dataframes, 2) put X and y df in the session state, 3) get their names and put both in session state
        # 4) write X and y df to csv file in data\ dir
        X_in_process_df = df_initial.drop(columns = y_column)
        st.session_state['X_in_process_df'] = X_in_process_df
        dir_and_X_IN_PROCESS_fname =  "data/" + "X_IN_PROCESS.csv"
        st.session_state['dir_and_X_IN_PROCESS_fname'] = dir_and_X_IN_PROCESS_fname
        X_in_process_df.to_csv(dir_and_X_IN_PROCESS_fname, index=False)

        # Now the y. Define, add to session state, get name and write to csv
        y_df = df_initial[[option]]
        st.session_state['y_df'] = y_df
        dir_and_y_fname =  "data/" + "y_IN_PROCESS.csv"
        y_df.to_csv(dir_and_y_fname,index=False)

        st.session_state['new_file'] = False
        # df_loaded will cover the x and y dataframes
        st.session_state['df_loaded'] = True
        st.session_state['X and y loaded'] = True

        #this can go latter
        with container_1:
            st.write("The y column is: {y_column}")
            st.text(f"type of y_df is: {type(y_df)}")
            st.write("X df is...")
            st.dataframe(X_in_process_df)
            st.write("y df...")
            st.dataframe(y_df)
            st.text('df_loaded is True and X_in_process_df is loaded')
            # now save the in process X dataframe. First get the path and name
            st.write(f"""## Files loaded. X and y dataframes are ready.
                 Proceed to 'Preprocessing' """)
       
        y_select_container.empty()
        
  
if 'df_initial_loaded' in st.session_state and st.session_state['df_initial_loaded']:
    # df.info() does not return anything to python, so need to pipe it
    with st.popover("Summary stats for loaded dataframe"):
        st.write(f"Summary for file {file_name} (dataframe: df_initial)")
        buffer = io.StringIO() 
        df_initial.info(buf=buffer)
        s = buffer.getvalue()  
        st.text(f"#### {s}")
        st.text("\n")
        st.write(f"The shape is: \n{df_initial.shape}")
        st.text(f"dataframe df_initial value_counts...")
        for the_col in df_initial.columns:
            st.text(f"dataframe df_initial value_counts are...\n{df_initial[the_col].value_counts()}")
   

if 'X and y loaded' in st.session_state and st.session_state['X and y loaded']:
    if st.button(":blue[READY: Proceed to 'Preprocessing?] Click to proceed"):
        st.switch_page(("pages/preprocessor.py"))
    st.write("Can also do some Exploratory Data Analysis / Graphing in 'EDA and Graphs' page")
    if st.button(":green[Click Here] for EDA and Graphs"):
        st.switch_page(("pages/EDA_and_graphs.py"))
