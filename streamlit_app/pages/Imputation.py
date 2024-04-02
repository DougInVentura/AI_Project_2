#streamlit run 'Main Page for Pipey.py'

# Encoder tool: 

# Data Used and Generated:
#   Uses:
#       currently none used. Data is defined in the MAIN section. 
#   Generates:
#       df_just_before_imputing.csv
#       final_imputed_df.csv

import streamlit as st
import numpy as np
import pandas as pd
import pandas.api.types as pd_api_types

# ------------------------------------------------------------
# this app uses 'Streamlit' for simple web development.
# references to st. are streamlit objects

DROP_NA = 1
MEAN_VALUE = 2
MODE_VALUE = 3
CUSTOM_VALUE = 4

#Default Imputation 1 = drop_na
DEFAULT_IMPUTATION_VALUE = DROP_NA

st.set_page_config(
    layout="wide",
    page_title="👋 Select Encoding Screen 👋",
    page_icon="👋"
)

st.write("""### Project 2 ASU AI Course: Supervised Machine Learning Classification App for Preprocessing, Encoding, Model Building and Scoring
#### Page: **Imputation**
#### This page will create the imputation strategy, and impute the data.
         
##### Page is using the .csv selected from 'Select Data...' screen
___________________________________________________________________________________________________________________________________
""")

def create_default_imputation_list(df):
    imputation_list = []
    for temp_col in df.columns:
        imputation_list.append(DEFAULT_IMPUTATION_VALUE)
    return imputation_list

def create_missing_percent_list(df):
    percent_missing_list = []
    for temp_col in df.columns:
        current = df[temp_col]
        percent_missing = current.isna().sum()/len(current)
        percent_missing_list.append(percent_missing)
    return percent_missing_list

def create_impute_strategy_dict(df):
    impute_dict = {}
    for col_name in df['column_names']:
        impute_dict[col_name] = df.loc[df['column_names'] == col_name, 'impute_num'].item()
    return impute_dict

def create_custom_value_dict(df, impute_strategy_dict):
    custom_value_dict = {}
    for col_name in df['column_names']:
        if impute_strategy_dict[col] == CUSTOM_VALUE:
            custom_value_dict[col_name] = df.loc[df['column_names'] == col_name, 'custom_value'].item()
    return custom_value_dict

def impute_data(x_data, x_train, impute_strategy_dict, custom_value_dict):
    for col in x_data.columns:
        impute_strategy = impute_strategy_dict[col]

        if impute_strategy == MEAN_VALUE:
            if pd_api_types.is_numeric_dtype(x_train.dtypes[col]):
                mean = x_train[col].mean()
            else:
                mean = str(np.NaN)
            x_data[col] = x_data[col].fillna(mean)
        elif impute_strategy == MODE_VALUE:
            mode = x_train[col].mode().iloc[0]
            x_data[col] = x_data[col].fillna(mode)
        elif impute_strategy == CUSTOM_VALUE:
            custom_value = custom_value_dict[col]
            x_data[col] = x_data[col].fillna(custom_value)
        #any other value (so 1 if they followed instructions) drop na since its the default
        else:
            x_data = x_data.dropna(subset=[col])
    return x_data

if ('train_test_loaded' in st.session_state) and (st.session_state['train_test_loaded']):
    print("train test loaded")

    if 'X_train' in st.session_state:
        x_train = st.session_state['X_train']
        print("got xtrain")
    else: 
        print('X_train does not exist')

    if 'X_test' in st.session_state:
        x_test = st.session_state['X_test']
        print("got xtest")
    else: 
        print('X_test does not exist')
   
    #show the dataframe
    st.dataframe(x_train.head(7))

    default_imputation_list = create_default_imputation_list(x_train)
    missing_percent_values_list = create_missing_percent_list(x_train)

    #set up the imputation df
    df_column_actions = pd.DataFrame({
        'column_names':x_train.columns,
        'data_types':x_train.dtypes,
        'percent_missing_values':missing_percent_values_list,
        'impute_num':default_imputation_list
    })
    df_column_actions = df_column_actions.sort_values(by='percent_missing_values', ascending=False).reset_index(drop=True)
    df_column_actions['custom_value'] = str(np.NaN)

    #Set up the instructions for specifying the imputation
    st.write("""For each column name, enter one of the following in the Impute_Num field...\n 
    1  for  'Drop NA'
    2  for  'Mean Value'
    3  for  'Mode Value'
    4  for  'Custom Value' 
    For Custom Value, must enter the custom value in the custom_value column""")

    with st.popover("Open Mean Values"):
        st.markdown("Mean Values for Columns")
        mean_list = []
        for col in x_train.columns:
            if pd_api_types.is_numeric_dtype(x_train.dtypes[col]):
                mean_list.append(x_train[col].mean())
            else:
                mean_list.append(str(np.NaN))

        mean_df = pd.DataFrame({
            'column_names':x_train.columns,
            'mean_values':mean_list
        })
        st.dataframe(mean_df)

    with st.popover("Open Mode Values"):
        st.markdown("Mode Values for Columns")
        mode_list = []
        for col in x_train.columns:
            mode_list.append(x_train[col].mode().iloc[0])

        mode_df = pd.DataFrame({
            'column_names':x_train.columns,
            'mode_values':mode_list
        })
        st.dataframe(mode_df)
        
    #set up the data editor with the imputing dictionary
    edited_df = st.data_editor(df_column_actions)
        
    if st.button('Ready: Run Imputation'):
        impute_strategy_dict = create_impute_strategy_dict(edited_df)
        custom_value_dict = create_custom_value_dict(edited_df, impute_strategy_dict)

        with st.popover("Open impute strategy dict"):
            impute_dict_df = pd.DataFrame(impute_strategy_dict, index=[0])
            st.dataframe(impute_dict_df)
        
        with st.popover("Open custom value dict"):
            custom_value_dict_df = pd.DataFrame(custom_value_dict, index=[0])
            st.dataframe(custom_value_dict_df)
    
        x_train_imputed = impute_data(x_data=x_train, x_train=x_train, impute_strategy_dict=impute_strategy_dict, custom_value_dict=custom_value_dict)
        x_test_imputed = impute_data(x_data=x_test, x_train=x_train, impute_strategy_dict=impute_strategy_dict, custom_value_dict=custom_value_dict)

        st.write("Check all values are filled")
        st.write("X_Train post imputation missing values:")
        st.dataframe(x_train_imputed.isna().sum()/len(x_train_imputed))
        st.dataframe(x_train_imputed)
        st.write("X_Test post imputation missing values:")
        st.dataframe(x_test_imputed.isna().sum()/len(x_test_imputed))
        st.dataframe(x_test_imputed)

        st.session_state['is_imputation_complete'] = True
        st.session_state['x_train_imputed'] = x_train_imputed
        st.session_state['x_test_imputed'] = x_test_imputed

        st.write("All done imputing go to 'Use Selected Encoding Steps'")