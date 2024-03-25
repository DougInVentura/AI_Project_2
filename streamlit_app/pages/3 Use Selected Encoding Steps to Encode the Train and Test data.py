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

def do_OHE_encoding(df, column_list):
    '''
        function performs one hot encoding. 
        Arguments   : df -> the dataframe of interest
                    : column_list -> list of columns in df to do one hot encoding on
        Returns     : Dataframe that has underwent one hot encoding.  column names are original names, underscore, column value. these are from get_feature_names_out(column_list)
    '''
    # Create a OneHotEncoder instance
    encoder_OHE = OneHotEncoder()  # could not specify dense_output, nor sparse=False. Got an error
    encoded_data = encoder_OHE.fit_transform(df[column_list]).toarray()
    ohe_feature_names = encoder_OHE.get_feature_names_out(column_list)
    df_OHE = pd.DataFrame(encoded_data, columns = ohe_feature_names)  # columns = feature_names_out  , columns = encoder_OHE.get_feature_names_out(column_list)
    return df_OHE

# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
def do_Ordinal_Encoding(df,column_list):
    '''
    Does ordinal encoding for specified columns.
    Arguments: df -> this is the entire dataframe you are encoding
             : column_list -> these are columns you want to do Ordinal Encoding
    Returns: encoded dataframe where Ordinal Encoding applied to each column
    Internal Notes: In a loop, displays each column name and unique values. User input is the order of the unique values, which is needed to do Ord Encoding correctly.
    '''
    os.system("cls")
    # df_for_ord_e is a dataframe containing only the columns you want to do Ord Encoding. 
    df_converted = df[column_list]
    ord_list = [] # likely not needed
    print("For Ordinal Encoding, we need the order of the values in each column")
    print("For each column, program will show the unique values, you then input the integer order from 1 to n separated by spaces")
    print("Example for Education: if the order of the value counts are: high school, primary, postgrad, college, you would enter 2 1 4 3")
    junk = input("press enter to continue")
    os.system("cls")
    # make a starter dataframe to build on. the "0th" column will be used.
    # in the end, df_converted is the ordinal encoded dataframe with only the selected columns from function param column_list
    # df_converted = df_for_ord_e.iloc[:,0].copy()
    # print(f"outside of loop, df_converted is: \n{df_converted}\n")
    for i, the_col in enumerate(column_list):
        uniq_levels = df_converted[the_col].unique()
        print("----------------")
        print(f"For column {the_col}, the unique values are:\nd{uniq_levels}")
        order_str = input("enter order string (integers 1 through N:  ")
        order_list = order_str.split()
        # print(f"order list: {order_list} ")
        ord_df = pd.DataFrame({'categories':uniq_levels,'the_order':order_list})
        ord_sorted_df = ord_df.sort_values(by='the_order').reset_index(drop=True)
        
        ord_sorted_list = ord_sorted_df['categories'].to_list()
        # using pandas Categorical is much easier than the Ord Encoder, which requires multiple reshaping and multiple lines of code.
        encoded_data = pd.Categorical(df_converted[the_col], categories=ord_sorted_list, ordered=True).codes
        df_converted = df_converted.reset_index(drop=True)
        # trouble w standard sytax. using this as work around!
        df_converted[the_col] = encoded_data
        print(f"Sorted category order is:\n{ord_sorted_df}")
        junk = input("Hit enter to continue... ")
        os.system("cls")
    return df_converted
        
# -----------------------------------------------------------------------------------------------------------
def encode_df(df, encoder_dict):
    '''
    Funtion orchestrates encoding using OHE, Label Encoding, Ordinal Encoding and finally all columns go through Standard Scaling
    
    '''
    frames = []  # in the end, individual data frames for OHE, Lab Encoding, Ord Encoding and Standard Scaping are appended to frames for column merging using pd.concat
    for the_key in encoder_dict.keys():
        column_list = encoder_dict[the_key]
        match the_key:
            case 'OHE':
                # print(f"for OHE: list is: {column_list}")
                df_OHE = do_OHE_encoding(df, column_list )
                frames.append(df_OHE)
                # print("df_OHE AFTER OHE encoding ...")
                # print(df_OHE)
            case 'LE':
                # print(f"for LE: list is: {column_list}")
                encoder_LE = LabelEncoder()
                df_encoded_LE = df[column_list].apply(lambda col: encoder_LE.fit_transform(col))
                # print(f"df_encoded_LE is: \n{df_encoded_LE}")
                frames.append(df_encoded_LE)
            case 'ORDE':
                # print(f"for ORDE: list is: {column_list}")
                df_Ordinal = do_Ordinal_Encoding(df,column_list)
                # print(f"for Ord Enc, the dataframe is: \n{df_Ordinal}")
                frames.append(df_Ordinal)
            case 'NS':
                # print(f"for NS only: list is: {column_list}")
                df_to_only_scale = df[column_list]
                frames.append(df_to_only_scale)
    df_to_scale = pd.concat(frames, axis = 1)
    df_to_scale.to_csv("df_just_before_scaling.csv")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_to_scale)
    df_scaled_final = pd.DataFrame(scaled_array, columns = df_to_scale.columns)
    return df_scaled_final

# ----------- MAIN   MAIN   MAIN ----------------------------------------
if st.session_state["df_loaded"] and st.session_state['Encoding_Dict_Ready']:
    st.write("OK we are ready to proceed")
else:
    st.write("not ready as yet!")