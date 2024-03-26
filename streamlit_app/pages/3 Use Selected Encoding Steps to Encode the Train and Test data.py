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

def do_OHE_encoding(df, column_list, encoding_dict):
    '''Function performs one hot encoding. 
        Arguments   : df -> the dataframe of interest
                    : column_list -> list of columns in df to do one hot encoding on
        Returns     : Dataframe that has underwent one hot encoding.  column names are original names, underscore, column value. these are from get_feature_names_out(column_list)
    '''
    # first create the max categories array
    encoder_OHE = OneHotEncoder()
    encoded_data = encoder_OHE.fit_transform(df[column_list]).toarray()
    ohe_feature_names = encoder_OHE.get_feature_names_out(column_list)
    df_OHE = pd.DataFrame(encoded_data, columns = ohe_feature_names)  # columns = feature_names_out  , columns = encoder_OHE.get_feature_names_out(column_list)
    return df_OHE

# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
def do_Ordinal_Encoding(df,column_list, encoding_dict):
    '''
    Does ordinal encoding for specified columns.
    Arguments: df -> this is the entire dataframe you are encoding
             : column_list -> these are columns you want to do Ordinal Encoding
    Returns: encoded dataframe where Ordinal Encoding applied to each column
    Internal Notes: In a loop, displays each column name and unique values. User input is the order of the unique values, which is needed to do Ord Encoding correctly.
    '''
    
    # it is called df_converted, but will converted through the loop below
    df_converted = df[column_list]
    for i, the_col in enumerate(column_list):
        uniq_levels = df_converted[the_col].unique()
        # the encoding_dict has the Ordinal encoding order for each affected column. Get it, sort the category string and set up to use pd.Categorical to encode the column.
        order_str = encoding_dict['ord_order_list'][i][the_col]
        order_list = order_str.split()
        ord_df = pd.DataFrame({'categories':uniq_levels,'the_order':order_list})
        ord_sorted_df = ord_df.sort_values(by='the_order').reset_index(drop=True)
        ord_sorted_list = ord_sorted_df['categories'].to_list()
        # using pandas Categorical is much easier than the Ord Encoder, which requires multiple reshaping and multiple lines of code.
        encoded_data = pd.Categorical(df_converted[the_col], categories=ord_sorted_list, ordered=True).codes
        df_converted = df_converted.reset_index(drop=True)
        # trouble w standard sytax. using this as work around!
        df_converted[the_col] = encoded_data
        
    # st.write("In do_Ordinal_Encding, 'df_converted' is as follows (after loop)")
    # st.dataframe(df_converted)
    return df_converted
        
# -----------------------------------------------------------------------------------------------------------
def encode_df(df, encoding_dict):
    '''
    Funtion orchestrates encoding using OHE, Label Encoding, Ordinal Encoding and finally all columns go through Standard Scaling
    Arguments   : df: This is the dataframe to encode
                : 
    '''
    # st.write(f"top of encode_df.  encoding_dict is: {encoding_dict}")
    frames = []  # in the end, individual data frames for OHE, Lab Encoding, Ord Encoding and Standard Scaping are appended to frames for column merging using pd.concat
    for the_key in encoding_dict.keys():
        column_list = encoding_dict[the_key]
        # Not all of the items in the encoding_dict need encoding. The column_list will be len() == 0
        if len(column_list) > 0:
            match the_key:
                case 'OHE':
                    df_OHE = do_OHE_encoding(df, column_list, encoding_dict)
                    # function returns dataframe which has One Hot Encoding applied
                    frames.append(df_OHE)
                case 'LabEnc':
                    # do label encoding
                    pdb.set_trace()
                    encoder_LE = LabelEncoder()
                    # Generate dataframe with with label encoding applied to select columns
                    df_encoded_LE = df[column_list].apply(lambda col: encoder_LE.fit_transform(col))
                    frames.append(df_encoded_LE)
                case 'OrdE':
                    df_Ordinal = do_Ordinal_Encoding(df,column_list,encoding_dict)
                    # Function above returns dataframe with Ordinal Encoding (as specified in the function) to the selected columns
                    frames.append(df_Ordinal)
                case 'NS':
                    # scaling not done at this point. BUT need those columns of the df for next steps below.
                    df_to_only_scale = df[column_list]
                    frames.append(df_to_only_scale)
                case 'max_categories_OHE':
                    message = 'do nothing. not in use'
                case 'max_categories_OHE' | 'current_date_time':
                    message = 'do nothing. not in use'
                case _:
                    st.text(f"reached case_ in encode_df. the_key is: {the_key}")
            
    # we now have all of the encoding pieces of the dataframe. they are in the 'frames' list. Concat and apply standard scaler
    df_to_scale = pd.concat(frames, axis = 1)
    st.write("df to scale created. Shown below... Will also archive in data dir as df_just_before_scaling.csv")
    st.dataframe(df_to_scale)
    df_to_scale.to_csv("df_just_before_scaling.csv")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df_to_scale)
    # scaled_array us numpy array. Convert to dataframe
    df_scaled = pd.DataFrame(scaled_array, columns = df_to_scale.columns)
    st.text("scaled dataframe...")
    st.dataframe(df_scaled)
    df_scaled.to_csv("df_encoding_and_scaling.csv")
    return df_scaled

# ----------- MAIN   MAIN   MAIN ----------------------------------------

if ('df_loaded' in st.session_state) and ('Encoding_Dict_Ready' in st.session_state) and st.session_state["df_loaded"] and st.session_state['Encoding_Dict_Ready']:
    st.write("Dataframe and Encoding Dictionary are Loaded... Proceeding")
    df_to_encode = st.session_state['df_in_process']
    st.session_state["df_to_encode"] = True
    encoding_dict = st.session_state["encoding_dict"]
    df_encoded_and_scaled = encode_df(df_to_encode, encoding_dict)
    st.session_state['is_df_scaled'] = True
    st.session_state['df_scaled'] = df_encoded_and_scaled
    st.write("**Ready to 'Run and Score Models'**")
else:
    st.write("Go back to 'Select Encoding Strategy' to select dataframe and build encoding dictionary")