from ast import main
from turtle import st
import pandas as pd
import os
import csv
import hvplot.pandas
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from Models.model import load_data, load_results, get_key_features, clean_data, create_X_y, split_data, predict_target, evaluate_model, save_model, save_keyfeatures, load_keyfeatures, save_plots

# import autosklearn.classification
from tpot import TPOTClassifier

# Initialize TPOT and let it optimize the ML pipeline
def optimize_pipeline(X_train, y_train):
    
    # Minimal version so it runs faster
    tpot = TPOTClassifier(generations=2, population_size=2, verbosity=2, random_state=42)
    
    #tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
    tpot.fit(X_train, y_train)
    return tpot

def predict_tpot(train_file_name, production_file_name, productionfile, model_name, y_field, overwrite = False):
    step = 1
    key_features = ''
    model_file = ''
    model_name = "TPOT"
    prod_or_train = "prod"

    try:
        
        # Load the train data
        df_train = load_data(production_file_name)
        
        # Load the production data
        df_prod = load_data(productionfile)
    
        # Clean the train data
        step = 2
        df = clean_data(df)
        
        # Clean the production data
        step = 3
        df_prod = clean_data(df_prod)
    
        # Create X and y
        step = 3
        X, y = create_X_y(df, y_field)
        
        # Create train X and y
        step = 3
        X_prod, y_prod = create_X_y(df_prod, y_field)
    
        # Use entire train dataset for training
        X_train = df_train.drop(columns=[y_field])
        y_train = df_train[y_field]
        
        # Production dataset becomes test dataset
        X_test = df_prod.drop(columns=[y_field])
        y_test = df_prod[y_field]        

        # Initialize TPOT and let it optimize the ML pipeline
        step = 6
        tpot = optimize_pipeline(X_train, y_train)
    
        # Predict the target values
        step = 7
        y_pred = predict_target(tpot, X_test)
        
        # Save the model
        step = 8
        save_model(production_file_name, prod_or_train, model_name, tpot.fitted_pipeline_)
            
        # Get key features
        step = 10
        key_features = get_key_features(tpot, X)
                
        #if key_features is None: then save X.columns as features
        if(key_features is None):
            key_features = X_prod.columns.tolist()
                
            save_keyfeatures(key_features, production_file_name,  prod_or_train, model_name)  
            
        else:
            # Load key features
            step = 11
            key_features = load_keyfeatures(production_file_name,  prod_or_train, model_name)
            
        # Evaluate the model - can't measure accuracy against production data
              
       
        step = 13
        
        # Add the predicted and Y values to the dataframe
        dataframe = pd.DataFrame(X_test)
        predicted_field = y_field + '_predicted'
        dataframe[predicted_field] = y_pred
        dataframe[y_field] = y_test   
      
        #Set overwrite to true while working to improve graphs
        #overwrite = True
        
        # Create a plot vs. y_field for each of the key features or X columns
        save_plots(production_file_name, prod_or_train, model_name, ['bar', 'line'], plot_folder, dataframe, key_features, predicted_field, y_field, overwrite)
    
        step = -1
        return {'error':step, 'accuracy': 0, 'key_features':key_features, 'model_file':model_file }
    except Exception as error:
        error += f" at step {step}"
        return {'error':error, 'accuracy': -1, 'key_features':'', 'model_file':'' }


# Main function
# Filename - data flie to be used
# Root folder for where plots will be saved
# y_field - the field to predict
# overwrite - if true, will overwrite existing plots (Used to force creation of new plots when the user has uploaded a newer version of the data file)
def train_tpot(filename, plot_folder, y_field, productionfile = None, overwrite = False):
    
    step = 1
    key_features = ''
    model_file = ''
    model_name = "TPOT"
    prod_or_train = "train"
    
    try:
        
        # Load the data
        df = load_data(filename)
    
        # Clean the data
        step = 2
        df = clean_data(df)
    
        # Create X and y
        step = 3
        X, y = create_X_y(df, y_field)
    
        # Split the data
        step = 4
        X_train, X_test, y_train, y_test = split_data(X, y)
    
        # Get Predictions from saved model or do the work required
        # so user doesn't have to wait for processing if already done and saved
        step = 5
        y_pred = load_results(filename, prod_or_train, model_name, X_test)
            
        if(y_pred is None):       
    
            # Initialize TPOT and let it optimize the ML pipeline
            step = 6
            tpot = optimize_pipeline(X_train, y_train)
    
            # Predict the target values
            step = 7
            y_pred = predict_target(tpot, X_test)
        
            # Save the model
            step = 8
            save_model(filename, prod_or_train, model_name, tpot.fitted_pipeline_)
            
            # Get key features
            step = 10
            key_features = get_key_features(tpot, X)
                
            #if key_features is None: then save X.columns as features
            if(key_features is None):
                key_features = X.columns.tolist()
                
            save_keyfeatures(key_features, filename, prod_or_train, model_name)  
            
        else:
            # Load key features
            step = 11
            key_features = load_keyfeatures(filename, prod_or_train, model_name)
            
        # Evaluate the model
        step = 12
        accuracy = evaluate_model(y_test, y_pred)
        
       
        step = 13
        
        # Add the predicted and Y values to the dataframe
        dataframe = pd.DataFrame(X_test)
        predicted_field = y_field + '_predicted'
        dataframe[predicted_field] = y_pred
        dataframe[y_field] = y_test   
      
        #Set overwrite to true while working to improve graphs
        #overwrite = True
        
        # Create a plot vs. y_field for each of the key features or X columns
        save_plots(filename, prod_or_train, model_name, ['bar', 'line'], plot_folder, dataframe, key_features, predicted_field, y_field, overwrite)
    
        step = -1
        return {'error':step, 'accuracy': accuracy, 'key_features':key_features, 'model_file':model_file }
    except Exception as error:
        error += f" at step {step}"
        return {'error':error, 'accuracy': -1, 'key_features':'', 'model_file':'' }
    
"""  These methods have beem relocated to models.py to be used by multiple models
# Load the data
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# clean the dat
def clean_data(df):
    df.dropna(inplace=True)
    df = pd.get_dummies(df, drop_first=False)
    return df

# create X and y
def create_X_y(df, y_field):
    y = df[y_field]
    X = df.drop(columns=[y_field])
    return X, y

# split the data
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test

# Predict the target values
def predict_target(tpot, X_test):
    y_pred = tpot.predict(X_test)
    return y_pred

# Evaluate the model
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def get_best_model(tpot):
    return tpot.fitted_pipeline_

#Look for a file with the same name as the data file but with a .config extension
#If it exists, load the settings from that file
#def load_results(fileName, X_test):    
    try:
#        # Load the saved model
        model_filename = fileName.replace('.csv', '.model')
        loaded_model = load(model_filename)

        # Use the loaded model for predictions
        return loaded_model.predict(X_test)
    except:
        return None

#def save_model(filename, tpot):
    # Save the best model to a file
    best_model = get_best_model(tpot)
    
    # create a filename based on the original filename but replace the extension with .model
    model_filename = filename.replace('.csv', '.model')
    dump(best_model, model_filename)
    return model_filename
    

#def create_save_plot(dataframe, filename, field1_name, field2_name):
    formatted_field1 = field1_name.replace('_', ' ').title()
    formatted_field2 = field2_name.replace('_', ' ').title()    
    dataframe.plot(kind='bar', x=field1_name, y=field2_name, rot=90, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}', grid=True)
    plt.savefig(filename)


#def save_plots(filename, model_name, plot_types, plot_folder, dataframe, key_features, y):
    # I admit this wouldn't work without allowing the user to select the field to use as the index
    try:
        
        #loop through the key features and create a plot for each
        for feature in key_features:
                        
            #skip encoded fields
            if(feature.find("uuid") != -1):
                continue
            
            for plot_type in plot_types:
                
                # Create the plot filename
                #new_folder = f"{os.path.split(filename)[0]}/{plot_folder}"
                current_directory = os.getcwd()
                new_folder = f"{current_directory}/{plot_folder}/{model_name}/plot_type"
                
                #Create folder if it doesn't exist
                os.makedirs(new_folder, exist_ok=True)
                
                # Build file name
                plot_filename = os.path.join(new_folder, os.path.basename(filename))
                plot_filename = plot_filename.replace('.csv', f"_{feature}_{plot_type}.png")
                
                # Create the plot if the file doesn't exist
                if not os.path.exists(plot_filename):
                    create_save_plot(dataframe, plot_filename, feature, y)                

    except Exception as error:
        Exception(f"Error generationg or saving plot(s) Error:{error}")


def get_key_features(model, X):
    # Get the key features
    try:
        feature_importances = model.best_pipeline.feature_importances_
        key_features = X.columns[feature_importances > 0.05]  # Adjust the threshold as needed
    
        return key_features
    except:
        Exception("Error getting key features")
    

def save_keyfeatures(key_features, filename):
    # Save the key features to a file
    key_features_filename = filename.replace('.csv', '.features')
    
    df = pd.DataFrame(key_features)
    df.to_csv(key_features_filename, index=False, header=False)

    return key_features_filename

def load_keyfeatures(filename):
    # Load the key features from a file
    key_features_filename = filename.replace('.csv', '.features')
    #if the file doesn't exist, return None
    if not os.path.exists(key_features_filename):
        return None
    
    key_features = pd.read_csv(key_features_filename)
    
    fiels_list = []
    # Add column 0 in each row to a list
    for i in range(len(key_features)):
         fiels_list.append(key_features.iloc[i, 0])
    
    
    return fiels_list
"""
  
