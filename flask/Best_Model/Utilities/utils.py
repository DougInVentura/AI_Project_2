from ast import main
import pandas as pd

"""
This file contains utility functions that are used by the application
"""

def get_columns(filename):
    """
    Load the data and return comma separated list of column names
    """
    df = pd.read_csv(filename)
    return ', '.join(df.columns)

def load_settings(fileName):
    """
    Look for a file with the same name as the data file but with a .config extension
    If it exists, load the settings from that file
    Settings will initially only contain the name of the column to predict (the 'y' value)
    """
    # Replace the csv extension with config
    fileName = fileName.replace('.csv', '.config')
    
    # Look for the file
    try:
        with open(fileName, 'r') as file:
            return file.read()
    except FileNotFoundError:
        pass
        
    return ""

def get_plot_file(filename):
    """
    Look for a file with the same name as the data file but with a .plot extension
    if found, return the name of the file
    """
    return filename.replace('.csv', '.png')

def save_settings(filename, fieldSelected):
    """
    Save the selected field to the session
    """
    # Replace the csv extension with config
    filename = filename.replace('.csv', '.config')
    
    # Create a text file with the selected field value
    with open(filename, 'w') as file:
        file.write(fieldSelected)
        
    return    



