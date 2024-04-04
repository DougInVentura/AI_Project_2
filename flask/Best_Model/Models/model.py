from ast import main
from turtle import st
from PIL.Image import ROTATE_90
from holoviews import Labels
import pandas as pd
import os
import csv
import hvplot.pandas
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
plt.rcParams["figure.figsize"] = (20,10)
sns.set_style('darkgrid')


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

#Look for a file with the same name as the data file but with a .config extension
#If it exists, load the settings from that file
def load_results(fileName, model_name, X_test):    
    try:
        # Load the saved model
        model_filename = fileName.replace('.csv', '')
        model_filename += f"_{model_name}.model"
        loaded_model = load(model_filename)
        
        # Use the loaded model for predictions
        return loaded_model.predict(X_test)
    except:
        return None


def save_model(filename, model_name, best_model):
    
    # if best_model is None, return None
    if best_model is None:
        return None
    
    # create a filename based on the original filename but replace the extension with .model
    model_filename = filename.replace('.csv', '')
    model_filename += f"_{model_name}.model"

    dump(best_model, model_filename)


def create_confusion_matrix(dataframe, filename, predicted, actual):

    contingency_table = pd.crosstab(dataframe[predicted], dataframe[actual])  
    sns.heatmap(contingency_table, annot=True, cmap="YlGnBu")  
    #plt.annotate = True
    plt.title('Confusion Matrix')

    classes = ['Negative', 'Positive']  # Class labels for binary classification

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    plt.clf()
    
""" backup
def create_confusion_matrix(dataframe, filename, predicted, actual):

    cm = confusion_matrix(dataframe[actual], dataframe[predicted])
    plt.clf()
    plt.annotate = True
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Negative', 'Positive']  # Class labels for binary classification

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    plt.clf()
"""

def create_save_plot(dataframe, filename, plot_type, field1_name, field2_name):
    formatted_field1 = field1_name.replace('_', ' ').title()
    formatted_field2 = field2_name.replace('_', ' ').title()
    
    unique_value_count = dataframe[field1_name].unique().size
    if unique_value_count > 10:
        num_bins = 10
        
        # Convert field1_name column to integer if it's a float
        if dataframe[field1_name].dtype == 'float64':        
            dataframe[field1_name] = dataframe[field1_name].astype(int)

        dataframe['Binned'] = pd.cut(dataframe[field1_name], bins=num_bins)
        grouped_data = dataframe.groupby('Binned').size()
        
        df = grouped_data.copy()
        print("Grouped Data")
        print(grouped_data.head())
        
        print("Dataframe")
        print(dataframe.head())
        
        df.groupby([field1_name, field2_name]).size().unstack().plot(kind=plot_type, stacked=True, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}')
        plt.xticks(rotation=90)   
        plt.tight_layout = True
        plt.savefig(filename)
        plt.clf()
   
    else:   
        
        df = dataframe.copy()
        df.groupby([field1_name, field2_name]).size().unstack().plot(kind=plot_type, stacked=True, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}')
        plt.xticks(rotation=90)        
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()


    """
    df = dataframe.copy()
    # Bar plot  
    df.groupby(['churn_predicted', 'churn']).size().unstack().plot(kind='bar', stacked=True)  
    plt.show()  

    
def create_save_plot(dataframe, filename, plot_type, field1_name, field2_name):
    formatted_field1 = field1_name.replace('_', ' ').title()
    formatted_field2 = field2_name.replace('_', ' ').title()
    
    unique_value_count = dataframe[field1_name].unique().size
    if unique_value_count > 10:
        num_bins = 10
        
        # Convert field1_name column to integer if it's a float
        if dataframe[field1_name].dtype == 'float64':        
            dataframe[field1_name] = dataframe[field1_name].astype(int)

        dataframe['Binned'] = pd.cut(dataframe[field1_name], bins=num_bins)
        grouped_data = dataframe.groupby('Binned').size()
        ax = grouped_data.plot(kind=plot_type, x='Binned', y=field2_name, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}', grid=False)
        ax.tick_params(axis='x', rotation=90)
        plt.tight_layout = True
        plt.savefig(filename)
   
    else:   
        
        dataframe.reset_index(drop=True, inplace=True)
        # print(dataframe.head())
        # dataframe.to_csv('dataframe.csv')
        # This code always produces ugly graphs - x labeling is unreadable and looks like many many values for x even if only a couple    
        # Y Horizontal , X vertical
        #dataframe.plot(kind=plot_type, x=field2_name, y=field1_name, rot=90, xlabel=formatted_field2, ylabel=formatted_field1, title=f'{formatted_field2} vs. {formatted_field1}', grid=True)
        # Y vertical, X horizontal
        ax = dataframe.plot(kind=plot_type, x=field1_name, y=field2_name, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}', grid=False)
        #ax.tick_params(axis='x', rotation=90)
        plt.xticks(rotation=90)        
        plt.tight_layout()
        plt.savefig(filename)
    """
    """ Backup
    def create_save_plot(dataframe, filename, plot_type, field1_name, field2_name):
    formatted_field1 = field1_name.replace('_', ' ').title()
    formatted_field2 = field2_name.replace('_', ' ').title()
    # Y Horizontal , X vertical
    #dataframe.plot(kind=plot_type, x=field2_name, y=field1_name, rot=90, xlabel=formatted_field2, ylabel=formatted_field1, title=f'{formatted_field2} vs. {formatted_field1}', grid=True)
    # Y vertical, X horizontal
    dataframe.plot(kind=plot_type, x=field1_name, y=field2_name, rot=90, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}', grid=False, plt.tight_layout)
    plt.tight_layout = True
    plt.savefig(filename)
    """

def create_hvplot(dataframe, filename, plot_type, field1_name, field2_name):
    formatted_field1 = field1_name.replace('_', ' ').title()
    formatted_field2 = field2_name.replace('_', ' ').title()
    plot = dataframe.hvplot.bar(x=field1_name, y=field2_name, rot=90, type=plot_type, xlabel=formatted_field1, ylabel=formatted_field2, title=f'{formatted_field1} vs. {formatted_field2}', grid=True)
    hvplot.save(plot, filename)
 

def save_plots(filename, model_name, plot_types, plot_folder, dataframe, key_features, predicted, actual_field, overwrite):
    # I admit this wouldn't work without allowing the user to select the field to use as the index
    try:
        
        #Create Confusion Matrix Plot if actual_field is not None
        if actual_field is not None:
            # Create the plot filename
            current_directory = os.getcwd()
            new_folder = f"{current_directory}/{plot_folder}/{model_name}"
            
            #Create folder if it doesn't exist
            os.makedirs(new_folder, exist_ok=True)
        
            # Build file name for confusion matrix
            cm_filename = os.path.join(new_folder, os.path.basename(filename))
            cm_filename = cm_filename.replace('.csv', f"_confusion_matrix.png")
            cm_filename = os.path.join(new_folder, os.path.basename(cm_filename))
            if not os.path.exists(cm_filename) or overwrite:
                create_confusion_matrix(dataframe, cm_filename, predicted, actual_field)

        #loop through the key features and create a plot for each
        for feature in key_features:
                        
            #skip encoded fields
            if(feature.find("uuid") != -1):
                continue
            
            for plot_type in plot_types:
                
                # Build the destination folder
                current_directory = os.getcwd()
                new_folder = f"{current_directory}/{plot_folder}/{model_name}"
                
                #Ensure destination folder exists (probably created above)
                os.makedirs(new_folder, exist_ok=True)
                
                # Build file name
                plot_filename = os.path.join(new_folder, os.path.basename(filename))
                plot_filename = plot_filename.replace('.csv', f"_{feature}_{plot_type}.png")
                #plot_filename = plot_filename.replace('.csv', f"_{feature}_{plot_type}.html")
                

                # Create the plot if the file doesn't exist
                if not os.path.exists(plot_filename) or overwrite:
                    create_save_plot(dataframe, plot_filename, plot_type, feature, predicted) 
                
                    #hvplot.save(plot, plot_filename)

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
    

def save_keyfeatures(key_features, filename, model_name):
    
    # Save the key features to a file
    key_features_filename = filename.replace('.csv', '')
    key_features_filename += f"_{model_name}.features"
  
    df = pd.DataFrame(key_features)
    df.to_csv(key_features_filename, index=False, header=False)

    return key_features_filename

def load_keyfeatures(filename, model_name):
    
    # Load the key features from a file
    key_features_filename = filename.replace('.csv', '')
    key_features_filename += f"_{model_name}.features"

    #if the file doesn't exist, return None
    if not os.path.exists(key_features_filename):
        return None
    
    key_features = pd.read_csv(key_features_filename)
    
    fiels_list = []
    
    # Add column 0 in each row to a list
    for i in range(len(key_features)):
         fiels_list.append(key_features.iloc[i, 0])    
    
    return fiels_list


