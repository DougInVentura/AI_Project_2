# Encoder tool: 

# Data Used and Generated:
#   Uses:
#       currently none used. Data is defined in the MAIN section. 
#   Generates:
#       df_just_before_scaling.csv
#       final_encoded_df.csv


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

def build_up_encoder_dict(df):
    ''' Function receives a dataframe. Loops over column names and allows user to specify Encoding for Machine Learning Applications
        Returns the encoded array, which is a list of dictionaries. The dictionary key is the column name and the 'value' is the encoding
    '''
    # for debuging, just return the finished list
    debug = False
    if debug:
        return {'OHE': ['animal'], 'LE': ['animal_name','breed'], 'ORDE': [], 'NS': ['rand_num', 'weight']}
    OHE = []
    LE = []
    ORDE = []
    NS = []
    # Need the columns and the column types for upcoming loop.
    # define them in a dataframe and sort by the object types ()
    df_columns_and_types = pd.DataFrame({'the_col':df.columns,'the_types':df.dtypes}).sort_values(by = 'the_types').reset_index(drop=True)
    encoder_dict_final = {}
    for i, the_col in enumerate(df_columns_and_types['the_col']):
        the_type = df_columns_and_types['the_types'][i]
        os.system('cls')
        print(f"Column: {the_col.upper()} -- Type: {the_type} -------------------------------------------------------------------------------------------------------------------------------------------")
        print("Enter 1 for OHE, 2 for Label Encoding, 3 for Ordinal Encoding, or 4 for numeric scaling for each column")
        print("-----------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("")
        print(f"Value Counts for {the_col}\n{df[the_col].value_counts()}\n")
        
        encoder_type = -1
        while (encoder_type < 1) or (encoder_type > 4):
            try:
                print("_____________________________________________")
                print("Enter 1 for OneHotE, 2 for LabelE(Label E), 3 for OrdE, or 4 for Num Scaling")
                print("_____________________________________________")
                encoder_type = int(input(f"Encoding for Column: '{the_col}':  "))
            except:
                print('enter int between 1 and 4')
        match encoder_type:
            case 1:
                encoder_code = 'OHE'
                OHE.append(the_col)
            case 2:
                encoder_code = 'LE'
                LE.append(the_col)
            case 3:
                encoder_code = 'ORDE'
                ORDE.append(the_col)
            case 4:
                encoder_code = 'NS'
                NS.append(the_col)
        os.system('cls')
        
    encoder_dict_all = {'OHE':OHE,'LE':LE,'ORDE':ORDE,'NS':NS}
    
    for the_key in encoder_dict_all.keys():
        if len(encoder_dict_all[the_key]) > 0:
            encoder_dict_final[the_key] = encoder_dict_all[the_key]
    print(f"encoder_dict_final is:\n{encoder_dict_final}")
    return encoder_dict_final

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

# -----------------------------------------------------------------------------------------------------------

def encode_and_scaled_df(df):
    encoder_dicts = build_up_encoder_dict(df)
    print("final encoder_list...")
    print(encoder_dicts)
    df_encoded = encode_df(df, encoder_dicts)  # does OHE, Label encoding, numeric
    return df_encoded


# -----------------------------------------------------------------------------------------------------------
# MAIN   MAIN   MAIN - starts here
# define a test df
# ----------------------------------------------------------------------------------------------------------

# define a test dataframe

df = pd.DataFrame({'animal':['cat', 'dog', 'rat', 'cat', 'dog'],
                   'animal_name':['ari','coco','willard','ava','bear'],
                   'rand_num':[10,20,30,40,50],
                   'weight': [15,30,1,16, 70],
                   'breed':['alley','mut','NYC rat','alley','golden'],
                   'education':['high school','college','primary','high school','post grad']})

os.system('cls')
print("---------- The Data --------------------------------------------------------------------------------------------------------------")
print(f"df head..\n{df.head()}")
print("----------------------------------------------------------------------------------------------------------------------------------")
junk = input("Press Enter to continue... ")

df_encoded = encode_and_scaled_df(df)
print(f"After encoding, the encoded df is: \n{df_encoded}")
df_encoded.to_csv("final_encoded_df.csv")