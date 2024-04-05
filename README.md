# ASU AI Course Project 2
## Project Team 3: Cody Cushing, David Gerhart, Doug Francis
## Date: 4/4/2024
## Project Title: Web Tool Development and Churn Analysis Project

## Intro
This 'Readme' file will descibe both the Streamlit Based Machine Learning Classification Tool for the web called "Pipey", as well as the High Level Flask-based tool implementing the TPOT and H20 AutoML algorithms.  Overviews and descriptions will be provided for both tools. In addition there are Repository Cloning and Usage instructions provided to run these models on any users machines. Prior to the overviews, descriptions and usage instructions, a file tree for this repository is provided.

## File Tree for this Repository

```
│   .gitignore
│   LICENSE
│   README.md
│
├───.vscode
│       launch.json
│
├───flask
│   │   Project 2 Flask Site possabilities.docx
│   │   Project 2 TPOT spike.ipynb
│   │
│   └───Best_Model
│       │   Best Model.pyproj
│       │   Best Model.pyproj.user
│       │   Best Model.sln
│       │   dataframe.csv
│       │   Ideas or Next Steps.txt
│       │   requirements.txt
│       │
│       ├───.vs
│       │   ├───Best Model
│       │   │   ├───config
│       │   │   │       applicationhost.config
│       │   │   │
│       │   │   ├───FileContentIndex
│       │   │   │       6565ef5d-9567-4e75-a496-5cac35be486a.vsidx
│       │   │   │       aa3b2ab6-06a3-4c3b-9fc7-ea4de5294180.vsidx
│       │   │   │       b171fbcb-e0ae-4a7c-a78b-74060d3b00f2.vsidx
│       │   │   │       bf798d53-4cdb-4161-86ef-7f62319cc8b0.vsidx
│       │   │   │       cc1f6ef8-a84c-4dc6-962b-cd88f5f202f2.vsidx
│       │   │   │
│       │   │   └───v17
│       │   │       │   .suo
│       │   │       │   DocumentLayout.json
│       │   │       │   fileList.bin
│       │   │       │
│       │   │       └───TestStore
│       │   │           └───0
│       │   │                   000.testlog
│       │   │
│       │   └───Project 2
│       │       └───v17
│       │               .suo
│       │               DocumentLayout.json
│       │
│       ├───Best_Model
│       │   │   runserver.py
│       │   │   views.py
│       │   │   __init__.py
│       │   │
│       │   ├───static
│       │   │   ├───content
│       │   │   │       bootstrap.css
│       │   │   │       bootstrap.min.css
│       │   │   │       site.css
│       │   │   │
│       │   │   ├───fonts
│       │   │   │       glyphicons-halflings-regular.eot
│       │   │   │       glyphicons-halflings-regular.svg
│       │   │   │       glyphicons-halflings-regular.ttf
│       │   │   │       glyphicons-halflings-regular.woff
│       │   │   │
│       │   │   └───scripts
│       │   │           bootstrap.js
│       │   │           bootstrap.min.js
│       │   │           jquery-1.10.2.intellisense.js
│       │   │           jquery-1.10.2.js
│       │   │           jquery-1.10.2.min.js
│       │   │           jquery-1.10.2.min.map
│       │   │           jquery.validate-vsdoc.js
│       │   │           jquery.validate.js
│       │   │           jquery.validate.min.js
│       │   │           jquery.validate.unobtrusive.js
│       │   │           jquery.validate.unobtrusive.min.js
│       │   │           modernizr-2.6.2.js
│       │   │           respond.js
│       │   │           respond.min.js
│       │   │           _references.js
│       │   │
│       │   └───templates
│       │           about.html
│       │           automl.html
│       │           contact.html
│       │           graph.html
│       │           index.html
│       │           layout.html
│       │           model3.html
│       │           result.html
│       │           select_file.html
│       │           tpot - Copy.html
│       │           tpot.html
│       │
│       ├───Models
│       │       AutoML.py
│       │       model.py
│       │       tpot.py
│       │
│       ├───obj
│       │   └───Any CPU
│       │       └───Debug
│       │               Best Model.pyproj.CoreCompileInputs.cache
│       │               Best Model.pyproj.FileListAbsolute.txt
│       │
│       ├───Resources
│       │       account_predict_data.config
│       │       account_predict_data.csv
│       │       churn_clean.config
│       │       churn_clean.csv
│       │
│       └───Utilities
│               utils.py
│
└───streamlit_app
    │   .gitignore
    │   df_encoding_and_scaling.csv
    │   df_just_before_scaling.csv
    │   gw_config.json
    │   Main Page for Pipey.py
    │   more_model_scoring.py
    │   streamlit_app.code-workspace
    │
    ├───archive
    │       non streamlit prototyping of code.ipynb
    │       test_streamlit_1.py
    │       test_streamlit_2.py
    │
    ├───data
    │       account_predict_data.csv
    │       churn.csv
    │       churn_clean.csv
    │       churn_max_cat_test.csv
    │       Encoding_Dictionary.txt
    │       RAW_test df animals3.csv
    │       test df animals.csv
    │       test df animals2.csv
    │       test df animals3.csv
    │       X_IN_PROCESS.csv
    │       y_IN_PROCESS.csv
    │
    └───pages
            1 Select data file and init proc.py
            2 preprocessor.py
            3 Select Encoding Strategy.py
            4 Imputation.py
            5 Use Selected Encoding Steps to Encode the Train and Test data.py
            6 Run and Score Models.py
            EDA_and_graphs.py
            pygwalker.py

```
## Part 1: "Pipey" - A Streamlit Based Machine Learning Classification Tool

## Overview and Description of "Pipey" - A Web Based Machine Learning Classification Tool
The name "Pipey" is a play on 'ML Pipelines'. The application provides custom data preparation from initial preprocessing, exploratory data analysis, various encoding and imputation strategies, and finally numeric scaling to prepare the data for running and scoring multiple classification models. The models used in the present version include:

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- AdaBoostClassifier
- SVC
- VotingClassifier

After the data processing which will be described in detail below, the application uses a ML Pipeline from Sklearn with six different classification models, all in-turn are then utilized by a "Voting Classifier" to provide the best prediction of the target variable, "Y," across the models.  The application is structured such that adding additional classification models is straightforward. More will follow below regarding the data preparation steps that has been built into version 1.0 of this application. For now, we will describe to use of the application.

### Web Interface
In order to make the application accessible and easy to use, the decision was made to create a web-based application. Initial prototyping using a standard terminal-based Python application made it clear that a web interface would be superior. To support the web development, a short informal evaluation of Django, Flash and Streamlit was carried out.  Note: This project includes a high-level tool which runs models such as TPOT that is not part of this Streamlit application and for that tool Flask was selected for use. This gave the team experience with both Streamlit and Flask, allowing a comparison of both to be carried out. For Pipey, Streamlit was ultimately selected due to the need for rapid development, where a lot of the web based complexity is handled by the Streamlit interface. Streamlit also had a relatively short learning curve and a large number of "widgets" and "components" to handle the user interface. Since data scientists are Streamlit’s intended use group, it is not surprising that it was well adapted to our requirements. 

### Data Preparation

As mentioned above, the application supports data preparation, which includes the typical steps needed to prepare the data for incorporation into supervised learning models. The data preparation steps include basic preprocessing, encoding, imputation, numeric normalization and partitioning of the data into training and testing data sets. Each of these steps will be discussed below:

#### Preprocessing:
The preprocessor screen supports a couple of basic preprocessing steps. These were designed to be used in conjunction with the graphing and exploratory data analysis section of the program as well.  At present, preprocessing only includes dropping columns and consolidating the number of categories that a feature has. Consolidating the number of categories is sometimes helpful with encoding techniques such as One Hot Encoding, when each new category can add an extra dimension to the dataset, which can cause issues.  

#### Encoding and Numeric Scaling:
The application supports various encoding approaches that the user can select on a feature-by-feature basis. This includes One Hot Encoding, Label Encoding and Ordinal Encoding. After encoding of all categorical columns, the encoding columns as well as the natively numeric columns in the data all undergo numeric scaling, since some of the models are based on minimizing distances, so getting all the data on the same scale is required. The scaling technique used is Z-Score Standardization, where each individual value is corrected by first subtracting the overall mean and then the resulting quantify is divided by the standard deviation of the data. This results in data with a mean of zero and a standard deviation of one but retains the shape of the initial distribution of the data.

#### Imputation
Following encoding and scaling, the application supports various imputation strategies. This includes dropping rows with missing values or filling in missing values with the overall mean of the data. Use of the mode or a custom value is also supported. 

#### Splitting Data into Training and Test Datasets.
After encoding, but prior to numeric scaling, the data is split into a training set and a test set. The default test versus train size built into Sklearn’s “train_test_split” is used in this application, which is 25% of the data is saved for testing and 75% of the data is used for training of the model. Splitting the data prior to numeric scaling and imputation keeps "data leakage" and bias from occurring. An explanation of this is beyond the scope of this write up.

#### Running and Scoring the Classification Models
After all of the data preparation steps which are described above are carried out and after the data is split into training and testing data sets, the models are fitted using the scaled training data (X_train). After model fitting, both the X_train and X_test data are transformed by the model into predicted response data for training and test. These are referred to as y_train_predicted and y_test_predicted. Since the actual “y” data for training and testing are known (y_train_actual and y_test_actual), the accuracy of the models can be calculated.  Pipey provides calculations of several scoring metrics including overall accuracy, balanced accuracy, and precision.  It also includes computation of the Confusion Matrix, the Classification Matrix, and Feature Importance graphs for each of the classification models.  This allows sufficient information to compare the models and to select the best performing models based on the scoring metrics of most interest to the data scientist.

#### Exploratory Data Analysis (EDA)
The application contains two different screens for EDA.  The first is called “EDA and graphs” and the second is called Pygwalker.  The EDA and Graphs screen provides basic scatter plotting of any of the features in the data set which are numeric. Options to color the points by or to size the points by are also provided. These can be used for both categorical and numeric features period. The screen is organized by tabs. The scatter plot is on the first tab. On the next tab histograms of any variable can be viewed. Finally, there is a final tab that provides the correlation matrix for all numeric columns in the data set.

#### Pygwalker
Pygwalker is a high-level exploratory data analysis tool that can be integrated into Python and Streamlit. Its feature set is similar to a basic implementation of Tableau or Spotfire. It has a drag and drop interface to select the X and Y columns, as well as various other abilities. This includes a variety of graphs for both the raw data or aggregations, faceting of the data, and representing features in the data set for coding the color, opacity, size, or shape of the data points. For more information see: https://docs.kanaries.net/pygwalker.


### How to build and run
Clone the Repo: On GitHub, look up https://github.com/DougInVentura/AI_Project_2. Clone the repository on your machine.  Ensure all of the appropriate Python libraries are installed.

**Libraries to Pip Install:** Use pip install to install Streamlit, Steamlit_extras, Pandas, Numpy, OS, IO, Plotly, Json, Datetime, sklearn and MatPlotLib.

**Running the Application:** Open the Streamlit_app folder in the clone of the repository on your machine in an editor such as Visual Studio Code. Open a Terminal window and ensure that it is pointing to the Streamlit_app directory. In the Terminal window, execute the command “streamlit run ‘Main Page for Pipey.py’”.

### Screen Flow and Processing.
Screen flow will progress as follows:
-	Main Page for Pipey
-	Select data File and Init Proc
-	Preprocessor
-	Select Encoding Strategy
-	Imputation
-	Use Selected Encoding Steps to Encode the Train and Test Data
-	Run and Score Models
Any time after loading the data (“select data file…” screen), you can also use the page navigation sidebar to perform some exploratory data analysis either using the “EDA and graphs” screen or the Pygwalker screen.

### Screen Flow
**‘Main Page for Pipey’:** From the “Main Page for Pipey” screen, click the button at the bottom of the screen. This will navigate to the “Select data File and Init Proc” screen.

**‘Select data File and Init Proc' Screen:** On this screen, click the ‘Browse files’ button to select a CSV file for processing. Once loaded, you’ll be asked to select the Y variable using a select box of the features in the dataset. A dropdown button to view some of the summary statistics for the dataset is also provided. Once a Y variable is selected, a button which is labeled “Ready to proceed to preprocessing. Click to proceed” (Note: a button to branch over and perform EDA is also provided, but this is not a required step). Clicking the proceed to preprocessing button will navigate to the preprocessor screen.

**‘Preprocessor’ screen:** This part of the workflow will allow you to drop columns that are not needed or to consolidate the categories in any categorical columns. After processing, click the button at the bottom of the page (it reads “Fill out instruction table, then Click when complete”). This will show the Y dataframe and the X dataframe. Then click the “Preprocessing is complete. Click here to go to ‘Select Encoding Strategy’”.

**'Select Encoding Strategy' screen:** On this screen, you can specify your encoding strategy. For numeric columns select "Numeric Scaling Only" (option 4). For each categorical column, select between One Hot encoding, Label encoding, or Ordinal Encoding.  For Ordinal encoding, use the drop-down button to view the Unique Values for the field. Note their order. Specify a numeric order in the "ordinal order" column. An example would be, if the Unique Values window for education shows High School, Primary, Graduate School and College, the ordinal order would be 2 1 4 3. this will be used to ensure that the numeric encoding of education follows the choosen order of Primary, High School, College and Graduate School. After the Encoding Instruction table is filled out. Click "Ready: make Encoding Dictionary". This will display the Encoding Strategy and Instructions you have selected. If correct, proceed by clicking the "Ready of Imputation. Click Here to go to Imputation" button.

**'Imputation' screen:** Fill out the table in the middle of the page using Options 1-4 to specify the imputation strategy you want to use for each column. If the percent_missing_values entry in the table is zero, it does not matter what option you select, since there are no missing values and no imputation will be performed for the column.  After filling out the 'Impute Num' field in the table and any custom values, click "Ready: Run Imputation". After clicking the button, new information will appear. It should indicate that each column should no longer contain any missing values in X_Train and X_test datasets.  At the bottom of the screen, click "Click Here to go to Use Selected Encoding Steps" to proceed.

**'Use Selected Encoding Steps to Encode the Train and Test Data' screen:** On this screen look over the X_train, X_test, y_train and y_test datasets in its encoded and scaled form. Note the intermediate files that have been saved in the 'streamlit_app/data' directory. If everything looks acceptable, click the "Click Here to go to Run and Score Models" button.

**'Run and Score Models' screen:** This screen runs on its own. For each model (such as logistic regression and decision trees, etc.), this screen will provide the accuracy obtained as well as a host of other scoring metrics for both the train and test datasets. The metrics provided include accuracy, balanced accuracy and precision. It also provides confusion matrices, classification reports and feature importance graphs.

**'EDA and Graphs' screen:** At any time after loading the data using the 'Select data File and Init Proc' screen, this screen can be selected from the sidebar navigation window. Follow the onscreen instructions to generate scatter plots for any features in your dataset. It also generates histograms and a correlation matrix for each of the X features in your dataset.

**'Pygwalker' screen:** After loading the data using the 'Select data File and Init Proc' screen, this screen can be selected from the sidebar navigation window. Use instructions for Pygwalker can be found at https://docs.kanaries.net/pygwalker.

_____

## Part II - High Level Flash Tool Implementing TPOT and H20 AutoML

### Flask Model Evaluation Tool
This tool allows non-technical people to perform machine learning.  Users can simply select a file; choose the field they want to predict, select a model they want to use, and click a ‘Begin processing’ button to perform the analysis.  Once the analysis is complete the customer will see the accuracy that was achieved, a list of fields that impact the score, and have the option to view graphs that visually represent the results.
To improve ease of use the program saves the name of the field that the user wants to predict. If that file is selected for review again the field to predict is auto selected.
To improve performance the results of the model are saved. If the file is looked at again these results will be loaded which eliminates the significant delay that often occurs when processing the data. All graphs and plots are also saved for later review without having to be rebuilt for display.
The program uses TPOT and AutoML to analyze the data. These tools handle the preprocessing for you so you don’t have to cleanup the data. That being said it’s **still very important** to remove data that will artificially cause the accuracy to be high. Examples of this would be any fields that represent the target field with a high degree of correlation.

### Packages required
```
from ast import main
import datetime
import filename
import h2o
from h2o.automl import H2OAutoML
from tpot import TPOTClassifier
from flask import render_template, request, session, send_file, send_from_directory
from flask.helpers import redirect
import joblib
import environ
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tkinter import SE
from turtle import st
import csv
import seaborn as sns
import hvplot.pandas
import matplotlib.pyplot
import os
import numpy as np
import pandas as pd
```

### How to build
- Tool Used Visual Studio 2022
- From Visual Studio Installer click 'Modify'
- Find and ensure Python development is selected (checked)
- In the lower right click Modify to begin installing any needed updates.
- I took the following steps to get started.
- Open visual Studio
- Click Create new project or solution
- Select Python as the language
- Search for Flask
- Select Flask Web Project
- Modify the project as needed for Project 2
- Select the conda environment you want to use
### How to run
- Once the required python features are installed (see above)
- Navigate to the flask\Best_Model folder
- Open Best Model.sln
- Select a python environment that has the required packages
- And run (F5)
### Screen Flow
#### Main Page:
This page describes the purpose of the application and provides high level information about the tools that are available to the user.  You are prompted to select a training file.  When you click on the button to select a training file the application takes you to a file selection page.
#### File Selection
This page is used to select the training file. Clicking on the select button lets you navigate anywhere on your system and choose a csv file. Once selected a list of columns is in the csv file is displayed and the user is prompted to choose the field they wish to predict.  Once the user verifies their choices they are taken back to the Main Page.
#### Main Page 2
Once back at the main page the user is shown the file and the column they wish to predict and are asked to choose a model to use.  Once they select a model then a page for the selected model is displayed.
#### Model Page
The model page describes the model they selected in further detail, and they are prompted to begin the training process.  Once the training process is complete the user is shown the accuracy of the prediction, a list of columns that impact the result and are given an opportunity to view line and bar graphs for each column along with a Confusion Matrix.

### Why Select Flask or Streamlit?

Flask and Streamlit offer different functionalities that can be utilized to achieve similar goals. Our research into the differences (and our experience using the two) suggest that Streamlit is better for quickly prototyping web apps focused on data science, while Flask offers much more customization in terms of page layout, content, and deployment[^1] while being more complicated to learn and requiring more coding overhead. Additionally, Flask is more robust in normal web applications since it allows for API requests while Streamlit does not[^2]. For what we were developing we wanted to test both approaches to discover what would work best for us in the future.

---------------------

### Work Cited

Any code used from internet sources are noted in the individual code files. An exception to this is that ChatGPT was used to generate suggestions for coding approaches at various points in the project.

[^1]: https://stackshare.io/stackups/flask-vs-streamlit#:~:text=Flask%20offers%20more%20control%20over,to%20the%20Streamlit%20sharing%20platform
[^2]: https://discuss.streamlit.io/t/why-using-flask-fastapi-when-there-is-streamlit/23615



