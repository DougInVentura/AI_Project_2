from ast import main
from tkinter import Y
from turtle import st
import pandas as pd
import os
import csv
import hvplot.pandas
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import Best_Model
from Models.model import load_data, load_results, get_key_features, clean_data, create_X_y, split_data, predict_target, evaluate_model, save_model, save_keyfeatures, load_keyfeatures, save_plots

# import autosklearn.classification
import h2o
from h2o.automl import H2OAutoML

# Initialize TPOT and let it optimize the ML pipeline
def optimize_pipeline(data, target_variable):
    
    print("Head of the data:")
    print(data.head())
    
    print("Description of the data:")
    print(data.describe())
    
    print("Columns:")
    print(data.columns)
    
    print("Shape of the data:")
    print(data.shape)
    
    print("Missing values:")
    print(data.isna().sum())
    
    # Clean the data (drop missing values))
    data = data.na_omit()

    print("Missing values after cleanup:")
    print(data.isna().sum())
    # Handle missing values (replace missing numerical values with mean, and missing categorical values with mode)
    #data.impute(method="mean", subset=["numerical_column"])
    #data.impute(method="mode", subset=["categorical_column"])

    # Encode categorical variables (one-hot encoding)
    #data = data.one_hot_encode("categorical_column")

    # Define predictors and response column
    # Not the last column predictors = data.columns[:-1]  # All columns except the last one (target variable)
    predictors = data.drop(target_variable)
    response = target_variable      # Name of the target variable column

    # Split data into training and validation sets
    train, test = data.split_frame(ratios=[0.8], seed=123)

    # Initialize AutoML
    aml = H2OAutoML(max_models=10, seed=123)

    # Train AutoML model
    aml.train(x=predictors, y=response, training_frame=train, validation_frame=test)

    #h2o_data =  h2o.H2OFrame(data)
   
    #train, test = h2o_data.split_frame(ratios=[0.8], seed=42)
    
    # Minimal version so it runs faster
    #aml = H2OAutoML(max_models=10, seed=42)
    #aml.train(x=x, y=y, training_frame=train)
    
    # View the leaderboard (list of models ranked by performance)
    lb = aml.leaderboard
    print(lb)
    
    

    #tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)
    return aml, test, predictors, response


def predict_automl(filename, y_field, plot_folder):
    return

# Main function
def train_automl(filename, plot_folder, y_field, productionfile = None, overwrite = False):
    
    step = 1
    key_features = ''
    model_file = ''
    model_name = "AutoML"
    prod_or_train = "train"
    test = None
    y_pred = None
    aml = None
    X = None
    y = None
    
    try:
        
        # Initialize H2O
        h2o.init()
        
        # Load the data
        #df = load_data(filename)
        df = h2o.import_file(filename)
    
        # Clean the data
        #step = 2
        #df = clean_data(df)
    
        # Create X and y
        #step = 3
        #X, y = create_X_y(df, y_field)
    
        # Convert the Pandas DataFrame to an H2O Frame
        #h2o_data = h2o.H2OFrame(df)
        
        if(y_pred is None):       
    
            # Initialize H20 AutoML and let it optimize the ML pipeline
            step = 6
            aml, test, X, y = optimize_pipeline(df, y_field)
                     
            #Get the best model
            step = 7
            best_model = aml.leader           
        
            # Save the best model
            step = 8
            model_file = save_model(filename, best_model)
            
            # Predict the target values
            y_pred = best_model.predict(df)
            
            # Get key features
            step = 10
            key_features = best_model.varimp()
            
                
            #if key_features is None: then save X.columns as features
            if(key_features is None):
                key_features = X.columns.tolist()
                
            save_keyfeatures(key_features, filename)  
            
        else:
            # Load key features
            step = 11
            key_features = load_keyfeatures(filename)
            
        # Evaluate the model
        step = 12
        predictions_df = y_pred.as_data_frame()
        actual_values = test[y].as_data_frame()
        accuracy = accuracy_score(actual_values, predictions_df)
        
        # Create interactive plot and save it to a file
        step = 13
        
        predicted_field = y_field + '_predicted'
        test[predicted_field] = y_pred

        # Create a plot vs. y_field for each of the key features or X columns
        save_plots(filename,  model_name, ['bar', 'line'], plot_folder, test, key_features, predicted_field)
    
        step = -1
        return {'error':step, 'accuracy': accuracy, 'key_features':key_features, 'model_file':model_file }
    except Exception as error:
        error += f" at step {step}"
        return {'error':error, 'accuracy': -1, 'key_features':'', 'model_file':'' }
    

  