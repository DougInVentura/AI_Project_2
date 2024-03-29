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

#Default Imputation 1 = drop_na
_DEFAULT_IMPUTATION_VALUE = 1


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
        imputation_list.append(_DEFAULT_IMPUTATION_VALUE)
    return imputation_list

def create_missing_percent_list(df):
    percent_missing_list = []
    for temp_col in df.columns:
        percent_missing_df = df.isna().sum()/len(x_train)
        percent_missing_df.iloc[:, :1]
        percent_missing_list.append(f"{}")
#X_train.isna().sum()/len(X_train)

if ('train_test_loaded' in st.session_state) and (st.session_state['train_test_loaded']):
    if 'X_train' in st.session_state:
        x_train = st.session_state['X_train']
    if 'X_test' in st.session_state:
        x_test = st.session_state['X_test']
   

    #show the dataframe
    st.dataframe(x_train.head(7))

    default_imputation_list = create_default_imputation_list(x_train)

    #set up the imputation df
    df_column_actions = pd.Dataframe({
        'column_names':x_train.columns,
        'data_types':x_train.dtypes,
        'percent_missing_values':
    })