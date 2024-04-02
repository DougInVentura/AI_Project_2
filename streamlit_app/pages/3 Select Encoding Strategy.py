
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
import time

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
         
##### Page is using the .csv selected from 'Select Data...' screen. It assumes Preprocessor page completed as well
___________________________________________________________________________________________________________________________________
""")

def get_default_encode_num(df):
    encode_list = []
    for the_col in df.columns:
        if df[the_col].dtype.name == 'object':
            encode_list.append(1)
        else:
            encode_list.append(4)
    return encode_list

if (('Ready_for_Imputation' not in st.session_state or st.session_state['Ready_for_Imputation'] == False)):
    if ('df_loaded' in st.session_state) and (st.session_state.df_loaded):
        st.session_state['Ready_for_Imputation'] = False  # at end of this block if all successful, then it becomes true
        df_initial = st.session_state['df_initial']
        X_in_process_df = st.session_state['X_in_process_df']
        y_df = st.session_state['y_df']

        st.write("### **X Dataframe** ")
        st.dataframe(X_in_process_df.head(7))
        # Need to code the 'object' columns with a 4 (numeric only) and code the non-object columns with a 1 (default is one hot encoding)
        default_encoding_list = get_default_encode_num(X_in_process_df)
        # Now set up the Encoding DF, which has the column names, the types, a place for the Encoding Number (type of encoding selected)
        df_column_actions = pd.DataFrame({'column_names':X_in_process_df.columns,'the_types':X_in_process_df.dtypes, "encode_num":default_encoding_list}).sort_values(by = 'the_types').reset_index(drop=True)
        df_column_actions['ordinal_order'] = str(np.NaN)
        # Now set up the instructions for specifying the Encoding
        st.write("""##### **For each column name, enter one of the following in the Encoded_Num field...** \n 
**1**  for  'One Hot Encoding' \n
**2**  for  'Label Encoding' \n
**3**  for  'Ordinal Encoding' or \n
**4**  for  'Numeric Scaling only' \n
For Ordinal Encoding, must also specify Ordinal Order **for each category** based on the unique entries (use popover button to view Unique column values""")

        with st.popover("Open Column Value Count Window"):
            st.markdown("Value Counts for Each Columns... 👋")
            for theCol in X_in_process_df.columns:
                st.write(X_in_process_df[theCol].value_counts())

        with st.popover("Open Unique Values Window (for Ordinal encoding)"):
            st.markdown("Unique values in each column... 👋")
            for theCol in X_in_process_df.columns:
                st.write(f"Column: {theCol}")
                st.write(X_in_process_df[theCol].unique())

        # Set up the data editor with the Encoding dictonary
        st.write("### **Encoding Instruction Table** ")
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
            ord_levels_for_col = []
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
                        ord_levels_for_col.append({the_col:list(X_in_process_df[the_col].unique())})
                    case 4:
                        NS.append(the_col)
                    case _:
                        what_u_thinkin.append(the_col)
            st.write(f"""**Encoding Strategy / Instructions**: \n
Numeric Scaling Only:        {NS}\n
Label Encoding:              {LabEnc}\n  
Ordinal Encoding is:         {OrdE}\n
One Hot Encoding (OHE):      {OHE}\n
Ordinal Levels for Columns:  {ord_levels_for_col}\n
Order for Ordinal Encoding:  {ord_order_list} """)  
            
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
                            'ord_levels_for_col': ord_levels_for_col,
                            'current_date_time':date_time_str}
            st.session_state['Encoding_Dict_Ready'] = True
            st.session_state['encoding_dict'] = encoding_dict
            with open("data/Encoding_Dictionary.txt", "w") as file:
                json.dump(encoding_dict, file)  # encode dict into JSON
            
            # train_test_split to get the X_train, X_test, y_train, y_test
            X_train, X_test, y_train, y_test = train_test_split(X_in_process_df, y_df, random_state=42)
            # reset the indexes
            X_train.reset_index(drop = True, inplace=True)
            X_test.reset_index(drop = True, inplace=True)
            y_train.reset_index(drop = True, inplace=True)
            y_test.reset_index(drop = True, inplace=True)
            
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['train_test_loaded'] = True

            st.write("#### Encoding Dictionary has been saved Ready for next step.'")
            st.session_state['Ready_for_Imputation'] = True

    else:   # df_Loaded either not in session_state or NOT true. Either way, user must go back to 'select daa file or preprocessing'
        st.write("### X, y and Initial dataframes not registering as loaded. Go back to either 'Select datafile...' or 'preprocessor'")
        st.session_state['train_test_loaded'] = False
        st.session_state['Ready_for_Imputation'] = False

if 'Ready_for_Imputation' in st.session_state and st.session_state['Ready_for_Imputation']:
    if st.button("Ready for 'Imputation. :blue[Click Here] to go to 'Imputation'"):
        st.switch_page("pages/4 Imputation.py")