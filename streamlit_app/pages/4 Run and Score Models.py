import streamlit as st

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pdb


if ('df_scaled' in st.session_state) and (st.session_state['is_df_scaled']):
    df_scaled = st.session_state['df_scaled']
    if len(df_scaled) > 0:
        st.write(f"Ready to go!  Scaled dataframe (df_scaled) beginning is below")
        st.dataframe(df_scaled.head(10))

    #TODO: remove this once Doug has created the data split session variables below
    #Split x and y 
    #test_df = pd.read_csv("data/account_predict_data.csv")
    #y_df = test_df['churn'] 
    #X_df = df_scaled.loc[:, df_scaled.columns != 'churn']
       
    
    #X_df = df_scaled.iloc[:,:-1]
    #y_df = df_scaled.iloc[:,-1:]

    #TODO: Doug will be doing the split elsewhere, so use that instead once it exists
    #Do the train test split
    #TODO: uncomment this once doug has created these variables
if ('are_X_frames__scaled' in st.session_state) and (st.session_state['are_X_frames__scaled']):
    if 'X_train_scaled' in st.session_state:
        X_train = st.session_state['X_train_scaled']

    if 'X_test_scaled' in st.session_state:
        X_test = st.session_state['X_test_scaled']

    if 'y_train' in st.session_state:
        y_train = st.session_state['y_train']

    if 'y_test' in st.session_state:
        y_test = st.session_state['y_test']
    

    #TODO: remove this once doug has created the session state for these variables
    #X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, random_state=42)
        
    #TODO: we might replace several of these with the lazy classifier since it does alot of comparrison work itself
    #Define/select which classifiers to use (For now just use the ones below)  
    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(random_state=42, max_depth=9)
    svc = SVC(probability=True, random_state=42)
    gbc = GradientBoostingClassifier(random_state=42)
    abc = AdaBoostClassifier(random_state=42) 

    # Create a VotingClassifier with soft voting
    voting_estimators = [
        ('lr', lr), 
        ('rf', rf), 
        ('svc', svc), 
        ('GBC', gbc),
        ('ABC',abc)]
    
    voting_clf = VotingClassifier(
        estimators = voting_estimators,
        voting = 'soft')
    
    classifier_list = []
    classifier_list.append(lr)
    classifier_list.append(rf)
    classifier_list.append(svc)
    classifier_list.append(gbc)
    classifier_list.append(abc)
    classifier_list.append(voting_clf)

    #TODO: Add steps to this pipeline for what we want to have
    #define the pipeline
    pipeline = Pipeline(steps=[
        ('classifier', VotingClassifier(
            estimators=voting_estimators,
            voting='soft'))
    ])

    #fit the pipeline
    pipeline.fit(X_train, y_train)
        
    #run the pipeline
    y_predictions = pipeline.predict(X_test)

    #TODO: REMOVE THIS PRINT, was for debugging purposes
    #print(f"Voting Classifier Accuracy: {(accuracy_score(y_test, y_predictions)*100):.2f}")

    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    #get the scores for each classifier
    classifier_score_string = ""
    for clf in classifier_list:
        clf.fit(X_train_transformed, y_train)
        y_test_predictions = clf.predict(X_test_transformed)

        #TODO: add this to a dataframe or format it to look pretty
        st.write(f"{clf.__class__.__name__:30}          Test Accuracy: {(accuracy_score(y_test, y_test_predictions)*100):.2f}    Test BALANCED Accuracy: {(balanced_accuracy_score(y_test, y_test_predictions)*100):.2f}")
        #classifier_score_string.join("\n")
    #total_score_string = "".join(classifier_score_string)
    st.write("All Done!")
    print("All Done!")

    #st.write(classifier_score_string)