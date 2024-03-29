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
import os
import streamlit as st
import pdb

st.set_page_config(
    layout="wide",
    page_title="👋 Apply Encoding Dictionary Screen 👋",
    page_icon="👋"
)

st.write("""### Page: **Use Selected Encoding Steps**
#### Page uses Enconding Dictionary and Selected Dataframe to produce Encoded and Scaled Dataframe""")

def do_OHE_encoding(X_train, X_test, column_list):
    '''Function performs one hot encoding. 
        Arguments   : X_train, X_test -> the dataframe of interest
                    : column_list -> list of columns in X_train and X_test to perform one hot encoding
        Returns     : Dataframe that has underwent one hot encoding.  column names are original names, underscore, column value. these are from get_feature_names_out(column_list)
    '''
    # first create the max categories array

    encoder_OHE = OneHotEncoder()
    X_all = pd.concat([X_train[column_list], X_test[column_list]], axis=0)
    encoder_OHE.fit(X_all[column_list])
    X_train_OHE_array = encoder_OHE.transform(X_train[column_list]).toarray()
    X_test_OHE_array = encoder_OHE.transform(X_test[column_list]).toarray()
    ohe_feature_names = encoder_OHE.get_feature_names_out(column_list)
    X_train_OHE_df = pd.DataFrame(X_train_OHE_array, columns = ohe_feature_names)  # columns = feature_names_out  , columns = encoder_OHE.get_feature_names_out(column_list)
    X_test_OHE_df = pd.DataFrame(X_test_OHE_array, columns = ohe_feature_names)
    return X_train_OHE_df, X_test_OHE_df

# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
def do_Ordinal_Encoding(X_train, X_test,column_list, encoding_dict):
    '''
    Does ordinal encoding for specified columns.
    Arguments: df -> this is the entire dataframe you are encoding
             : column_list -> these are columns you want to do Ordinal Encoding
    Returns: encoded dataframe where Ordinal Encoding applied to each column
    Internal Notes: In a loop, displays each column name and unique values. User input is the order of the unique values, which is needed to do Ord Encoding correctly.
    '''
    
    # it is called X_txxx_converted, but will converted through the loop below one column at a time
    X_train_converted = X_train[column_list].copy()
    X_test_converted = X_test[column_list].copy()
    
    for i, the_col in enumerate(column_list):
        uniq_levels = X_train[the_col].unique()
        # the encoding_dict has the Ordinal encoding order for each affected column. Get it, sort the category string and set up to use pd.Categorical to encode the column.
        order_str = encoding_dict['ord_order_list'][i][the_col]
        order_list = order_str.split()
        ord_df = pd.DataFrame({'categories':uniq_levels,'the_order':order_list})
        ord_sorted_df = ord_df.sort_values(by='the_order').reset_index(drop=True)
        ord_sorted_list = ord_sorted_df['categories'].to_list()
        # using pandas Categorical is much easier than the Ord Encoder, which requires multiple reshaping and multiple lines of code.
        X_train_Ord_col = pd.Categorical(X_train_converted[the_col], categories=ord_sorted_list, ordered=True).codes
        X_test_Ord_col = pd.Categorical(X_test_converted[the_col], categories=ord_sorted_list, ordered=True).codes
        
        # trouble w standard sytax. using this as work around!
        X_train_converted[the_col] = X_train_Ord_col
        X_test_converted[the_col] = X_test_Ord_col

    return X_train_converted, X_test_converted
        
# -----------------------------------------------------------------------------------------------------------
def encode_X_train_test(X_train, X_test, encoding_dict):
    '''
    Funtion orchestrates encoding using OHE, Label Encoding, Ordinal Encoding and finally all columns go through Standard Scaling
    Arguments   : X_train, X_test: This is the dataframe to encode
                : 
    '''
    # st.write(f"top of encode_X_train_test.  encoding_dict is: {encoding_dict}")
    X_train_frames = []  # in the end, individual data frames for OHE, Lab Encoding, Ord Encoding and Standard Scaping are appended to frames for column merging using pd.concat
    X_test_frames = []

    for the_key in encoding_dict.keys():
        column_list = encoding_dict[the_key]
        # Not all of the items in the encoding_dict need encoding. The column_list will be len() == 0
        if len(column_list) > 0:
            match the_key:
                case 'OHE':
                    X_train_OHE, X_test_OHE = do_OHE_encoding(X_train, X_test, column_list)
                    # function returns dataframe which has One Hot Encoding applied
                    X_train_frames.append(X_train_OHE)
                    X_test_frames.append(X_test_OHE)
                case 'LabEnc':
                    # do label encoding
                    encoder_LE = LabelEncoder()
                    # I am going to train on all the X's across X train and X test. It will not cause 'leakage' and it will ensure no missing categories / level
                    # in X_test
                    X_all = pd.concat([X_train[column_list], X_test[column_list]], axis=0)
                    encoder_LE.fit(X_all[column_list])
                    # Generate dataframe with with label encoding applied to select columns
                    
                    X_train_LE = X_train[column_list].apply(lambda col: encoder_LE.transform(col))
                    X_test_LE = X_test[column_list].apply(lambda col: encoder_LE.transform(col))
                    X_train_frames.append(X_train_LE)
                    X_test_frames.append(X_test_LE)
                case 'OrdE':
                    X_train_Ordinal, X_test_Ordinal = do_Ordinal_Encoding(X_train, X_test,column_list,encoding_dict)
                    # Function above returns dataframe with Ordinal Encoding (as specified in the function) to the selected columns
                    X_train_frames.append(X_train_Ordinal)
                    X_test_frames.append(X_test_Ordinal)
                case 'NS':
                    # scaling not done at this point. BUT need those columns of the df for next steps below.
                    X_train_to_only_scale = X_train[column_list]
                    X_test_to_only_scale = X_test[column_list]
                    X_train_frames.append(X_train_to_only_scale)
                    X_test_frames.append(X_test_to_only_scale)
                case 'max_categories_OHE':
                    message = 'do nothing. not in use'
                case 'current_date_time':
                    message = 'do nothing. not in use'
                case _:
                    st.text(f"reached case_ in encode_X_train_test. the_key is: {the_key}")
            
    # we now have all of the encoding pieces of the dataframe. they are in the 'frames' list. Concat and apply standard scaler
    X_train_to_scale = pd.concat(X_train_frames, axis = 1)
    X_test_to_scale = pd.concat(X_test_frames, axis = 1)
    st.write("X_train to scale created. Shown below... Also archived in data dir as X_train_before_scaling.csv")
    st.dataframe(X_train_to_scale)
    X_train_to_scale.to_csv("data/X_train_before_scaling.csv")
    # Now do the X_test_to_scale
    st.write("X_test to scale created. Shown below... Also archived in data dir as X_test_before_scaling.csv")
    st.dataframe(X_test_to_scale)
    X_train_to_scale.to_csv("data/X_test_before_scaling.csv")

    # Now do the standard scaler
    scaler = StandardScaler()
    X_train_scaled_array = scaler.fit_transform(X_train_to_scale)
    X_test_scaled_array = scaler.transform(X_test_to_scale)
    # scaler returns numpy array. Convert to dataframe
    #pdb.set_trace()
    X_train_scaled_df = pd.DataFrame(X_train_scaled_array, columns = X_train_to_scale.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled_array, columns = X_test_to_scale.columns)
    st.text("X_train_scaled dataframe...")
    st.dataframe(X_train_scaled_df)
    st.text("X_test_scaled dataframe...")
    st.dataframe(X_test_scaled_df)
    X_train_scaled_df.to_csv("data/X_train_scaled_df.csv", index = False)
    X_test_scaled_df.to_csv("data/X_test_scaled_df.csv", index = False)
    return X_train_scaled_df, X_test_scaled_df

# ----------- MAIN   MAIN   MAIN ----------------------------------------

if ('df_loaded' in st.session_state) and ('Encoding_Dict_Ready' in st.session_state) and 'train_test_loaded' in st.session_state \
        and st.session_state["df_loaded"] and st.session_state['Encoding_Dict_Ready'] and st.session_state['train_test_loaded']:
    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']
    st.write("Dataframe and Encoding Dictionary are Loaded... Proceeding")
    
    encoding_dict = st.session_state["encoding_dict"]
    
    X_train_scaled, X_test_scaled = encode_X_train_test(X_train, X_test, encoding_dict)
    st.session_state['are_X_frames__scaled'] = True
    st.session_state['X_train_scaled'] = X_train_scaled
    st.session_state['X_test_scaled'] = X_test_scaled
    st.write("**Ready to 'Run and Score Models'**")
else:
    st.write("Go back to 'Select Encoding Strategy' to select dataframe and build encoding dictionary")