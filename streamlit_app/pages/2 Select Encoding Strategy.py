
# Encoder tool: 

# Data Used and Generated:
#   Uses:
#       currently none used. Data is defined in the MAIN section. 
#   Generates:
#       df_just_before_scaling.csv
#       final_encoded_df.csv

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, add_dummy_feature, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import json
import os
import datetime
import pdb

# ------------------------------------------------------------
# this app uses 'Streamlit' for simple web development.
# references to st. are streamlit objects


st.set_page_config(
    layout="wide",
    page_title="👋 Select Encoding Screen 👋",
    page_icon="👋"
)


st.write("""### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
#### Page: **Select Enccoding Strategy**
#### This page will help you define the encoding dictonary for use in "Use Selected Encoding Steps..." page to encode the data.
         
##### Page is using the .csv selected from 'Select Data...' screen
___________________________________________________________________________________________________________________________________
""")

st.session_state["df_loaded"] = False
if 'df_in_process' in st.session_state:
    df_test_1 = st.session_state['df_in_process']
    st.session_state["df_loaded"] = True
else:
    uploaded_file = st.file_uploader("##### Choose the INIT_PROC_###.csv file from the data directory")
    if uploaded_file is not None:  
        df_test_1 = pd.read_csv(uploaded_file, index_col=False)  # get rid of the index)
        st.session_state['df_in_process'] = df_test_1
        st.session_state["df_loaded"] = True


if ('df_loaded' in st.session_state) and st.session_state["df_loaded"]:
    st.dataframe(df_test_1.head(7))
    # Now set up the Encoding DF, which has the column names, the types, a place for the Encoding Number (type of encoding selected)
    df_column_actions = pd.DataFrame({'column_names':df_test_1.columns,'the_types':df_test_1.dtypes}).sort_values(by = 'the_types').reset_index(drop=True)
    df_column_actions['encode_num'] = 1
    df_column_actions['ordinal_order'] = str(np.NaN)
    # Now set up the instructions for specifying the Encoding
    st.write("""For each column name, enter one of the following in the Encoded_Num field...\n 
    1  for  'One_Hot_Encoding'
    2  for  'Label Encoding
    3  for  'Ordinal Encoding or
    4  for  'Numeric Scaling only' 
    For OrdinalEncoding, must specify Ordinal Order based on the unique entries (use popover button to view Unique column values""")

    with st.popover("Open Column Value Count Window"):
        st.markdown("Value Counts for Columns... 👋")
        for theCol in df_test_1.columns:
            st.write(df_test_1[theCol].value_counts())

    with st.popover("Open Unique Values Window (for Ordinal encoding)"):
        st.markdown("Value Counts for columns... 👋")
        for theCol in df_test_1.columns:
            st.write(f"Column: {theCol}")
            st.write(df_test_1[theCol].unique())

    # Set up the data editor with the Encoding dictonary
    edited_df = st.data_editor(df_column_actions)

    # Process click of 'Ready... ' button
    if st.button('Ready: Make Encoding Dictonary'):
        # these are the encoding lists which will have the columns that need the different processing types.
        st.session_state['Encoding_Table_Ready'] = True
        OHE = []
        LabEnc = []
        OrdE = []
        NS = []
        ord_order_list = []
        # Which encoding is used list - from the dataframe edits user does
        which_encoding = []

        what_u_thinkin = []  # junk category. If this is not null, should trap and redo the matrix
        for index, row in edited_df.iterrows():
            # Access cell values in each row
            the_col = row['column_names']
            encoding_num = row['encode_num']
            ord_order = row['ordinal_order'] 
            # Process cell values as needed
            
            match encoding_num:
                case 1:
                    OHE.append(the_col)
                case 2:
                    LabEnc.append(the_col)
                case 3:
                    OrdE.append(the_col)
                    ord_order_list.append({the_col:ord_order})
                case 4:
                    NS.append(the_col)
                case _:
                    what_u_thinkin.append(the_col)
        st.write(f"""Column encoding specified: \n
        Numeric Scaling Only:    {NS}\n
        Label Encoding:          {LabEnc}\n  
        Ordinal Encoding is:     {OrdE}\n
        One Hot Encoding (OHE):  {OHE}\n
        Order for Ordinal E:     {ord_order_list} """)  
        
        if len(OHE) > 0:
            which_encoding.append('OHE')
        if len(NS) > 0:
            which_encoding.append('NS')
        if len(LabEnc) > 0:
            which_encoding.append('LabEnc')
        if len(OrdE) > 0:
            which_encoding.append('OrdE')
        date_time_str = str(datetime.datetime.now())
    
        encoding_dict = {'which_encoding_list': which_encoding,
                        'OHE':OHE,
                        'NS':NS,
                        'LabEnc':LabEnc,
                        'OrdE':OrdE,
                        'ord_order_list': ord_order_list,
                        'current_date_time':date_time_str}
        st.session_state['Encoding_Dict_Ready'] = True
        st.session_state['encoding_dict'] = encoding_dict
        with open("data/Encoding_Dictionary.txt", "w") as file:
            json.dump(encoding_dict, file)  # encode dict into JSON
        st.write("### Encoding Dictionary has been saved Ready for next step. Now go to 'Use Selected Encoding Steps...'")
        