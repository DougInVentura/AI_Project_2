# AI_Project_2
## Project Team 3: Cody Cushing, David Gerhart, Doug Francis
## Date: 4/4/2024

## Intro
This read me will descibe both the Streamlit Based Machine Learning Classification Tool for the web called "Pipey" and the High Level Flask-based tool implementing the TPOT algorithm.  Overviews and descriptions will be provided for both tools. In addition there are Repository Cloning and Usage instructions. 


## Overview and Description of "Pipey" - A Web Based Machine Learning Classification Tool
The name "Pipey" is a play on 'ML Pipelines'. The application provides custom data preparation from initial preprocessing, exploratory data analysis, various encoding and imputation strategies, and finally numeric scaling to prepare the data for running and scoring multiple classification models. The models used in the present version include:

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- GradientBoostingClassifier
- AdaBoostClassifier
- SVC
- VotingClassifier

After the data processing which will be described in detail below, the application uses a ML Pipeline from Sklearn with six different classification models, all in-turn are then utilized by a "Voting Classifier" to provide the best prediction of their target ("Y") variable across the models.  The application is structured such that adding additional classification models is straight forward. More will follow below regarding the data preprocessing that has been built into version 1.0 of this application. For now, we will pivot to usability of the application.

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
The application contains two different screens for EDA.  The first is called “EDA and graphs” and the second is called Pygwalker.  The EDA and Graphs screen provides basic scatter plotting of any of the features in the data set which are numeric. Options to color the points by or to size the points by are also provided. These can be used for both categorical and numeric features period. The screen is organized by tab tabs. The scatter plots on the first tab. On the next tab histograms of any variable can be viewed. Finally, there is a final tab that provides the correlation matrix for all numeric calms and the data set.

#### Pygwalker
Pygwalker is a high-level exploratory data analysis tool that can be integrated into Python and streamlet. Its feature set is similar to a basic implementation of Tableau or Spotfire. It has a drag and drop interface to select the X and Y columns, as well as various other abilities. This includes a variety of graphs for both the raw data or aggregations, faceting of the data, and representing features in the data set for coding the color, opacity, size, or shape of the data points. For more information see: https://docs.kanaries.net/pygwalker.


### How to build and run
Clone the Repo: On GitHub, look up https://github.com/DougInVentura/AI_Project_2. Clone the repository on your machine.  
Libraries to Pip Install: Use pip install to install Streamlit, Steamlit_extras, Pandas, Numpy, OS, IO, Plotly, Json, Datetime, sklearn and MatPlotLib.
Running: Open the Streamlit_app folder in the clone of the repository on your machine in an editor such as Visual Studio Code. Open a Terminal window and ensure that it is pointing to the Streamlit_app directory. In the Terminal window, execute the command “streamlit run ‘Main Page for Pipey.py’”.
Screen Flow and Processing.  Screen flow will progress as follows:
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

**‘Select data File and Init Proc' Screen:** On this screen, click the ‘Browse files’ button to select a CSV file for processing. Once loaded, you’ll be asked to select the Y variable using a select box of the features in the dataset. A drop down / pop up button to view some of the summary statistics for the dataset is also provided. Once a Y variable is selected, a button appears indicate “Ready to proceed to preprocessing. Click to proceed” (Note: a button to branch over and perform EDA is also provided, but this is not a required step). Clicking the “proceed to preprocessing” button will navigate to the preprocessor screen.

**‘Preprocessor’ screen:** This part of the workflow will allow you to drop columns that are not needed or to consolidate the categories in any categorical columns. After processing, click the button at the bottom of the page (it reads “Fill out instruction table, then Click when complete”. This will show the Y dataframe and the X dataframe. Then click the “Preprocessing is complete. Click here to go to ‘Select Encoding Strategy’”.

**'Select Encoding Strategy' screen:** On this screen, you can specify your encoding strategy. For numeric columns select "Numeric Scaling only" (option 4). For each categorical column, select between One Hot encoding, Label encoding, or Ordinal Encoding.  For Ordinal encoding, use the drop-down button to view the Unique Values for the field. Note their order. Specify a numeric order in the "ordinal order" column. An example would be, if the Unique Values window for education shows High School, Primary, Graduate School and College, the ordinal order would be 2 1 4 3. this will be used to ensure that the numeric encoding of education follows a sorted order of Primary, High School, College and Graduate School. After the Encoding Instruction table is filled out. Click "Ready: make Encoding Dictionary". This will display the Encoding Strategy and Instructions you have selected. If correct, proceed by clicking the "Ready of Imputation. Click Here to go to Imputation" button.

**'Imputation' screen:** Fill out the table in the middle of the page using Options 1-4 to specify the imputation strategy you want to use for each column. If the percent_missing_values entry in the table is zero, it does not matter what option you select, since there are no missing values and no imputation will be performed for the column.  Afte filling out the Impute Num field in the table and any custom values, click "Ready: Run Imputation". IN the data below, each column should not indicate that there are no remaining missing values in X_Train and X_test (the data has been split into train and test behind the scenes at this point).  At the bottom of the screen, click "Click Here to go to Use Selected Encoding Steps".

**'Use Selected Encoding Steps to Encode the Train and Test Data' screen:** On this screen look over the X_train, X_test, y_train and y_test data in its encoding form and scaled form. Note the intermediate files that have been saved in the 'streamlit_app/data' directory. If everything looks acceptable, click the "Click Here to go to Run and Score Models" button.

**'Run and Score Models' screen:** This screen runs on its own. for each model (such as logistic regression and decision trees, etc.), this screen will provide the accuracy obtained as well as a host of other scoring metrics for both the train and test datasets. The metrics provided include accuracy, balanced accuracy, precision, a confusion matrix, a classification report and graphs of feature importance.

**'EDA and Graphs' screen:** After loading the data using the 'Select data File and Init Proc' screen, this screen can be selected from the sidebar navigation window. Follow the onscreen instructions to generate scatter plots for features or your Y attribute, along with histograms and Correlation Matrices for each of the X features in your dataset.

**'Pygwalker' screen:** After loading the data using the 'Select data File and Init Proc' screen, this screen can be selected from the sidebar navigation window. Use instructions provide at https://docs.kanaries.net/pygwalker.

### Why use Flask and Streamlit

Flask and Streamlit offer different functionalities that can be utilized to achieve similar goals. Our research into the differences (and our experience using the two) suggest that Streamlit is better for quickly prototyping web apps focused on data science, while Flask offers much more customization in terms of page layout, content, and deployment[^1] while being more complicated to learn and requiring more coding overhead. Additionally, Flask is more robust in normal web applications since it allows for API requests while Streamlit does not[^2]. For what we were developing we wanted to test both approaches to discover what would work best for us in the future.

[^1]: https://stackshare.io/stackups/flask-vs-streamlit#:~:text=Flask%20offers%20more%20control%20over,to%20the%20Streamlit%20sharing%20platform
[^2]: https://discuss.streamlit.io/t/why-using-flask-fastapi-when-there-is-streamlit/23615


-------------------------------------------------------------------



Note: 

    How to build and use (includes user instructions)

#### David on Flask
    Technologies    
    How to build and use (includes user instructions)

### File tree
    David on flash app
    Cody on Streamlit


### Work cited
    Flash
    Streamlit - Doug

### Flask versus Streamlit
    David


Notes for presentation
Live demo
David on Flask
Cody: Streamlit up through imputation
Doug: Streamlit after imputation

Remaining Tasks:
* Readme updates per description above
* Practice presentations
* David - presentation power point draft for us to review on Wednesday
* David to add two more more models



