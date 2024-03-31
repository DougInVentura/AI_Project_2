import streamlit as st

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, roc_auc_score
import pdb


if ('Ready_to_Run_and_Score' in st.session_state) and (st.session_state['Ready_to_Run_and_Score']):
    X_train_scaled = st.session_state['X_train_scaled']
    X_test_scaled = st.session_state['X_test_scaled']
    y_train = st.session_state['y_train']
    y_test = st.session_state['y_test']

    col1, col2 = st.columns([1,4])
    with col1:
        st.write(f"#### **y_train**")
        st.dataframe(y_train.head(10))
        st.write("\n\n")
        st.write(f"#### **y_test**")
        st.dataframe(y_test.head(10))

    with col2:
        st.write(f"#### **X_train_scaled**")
        st.dataframe(X_train_scaled.head(10))
        st.write("\n\n")
        st.write(f"#### **X_test_scaled**")
        st.dataframe(X_test_scaled.head(10))


    st.write(f"#### Ready to go. Proceeding")



    # I asked ChatGPT 4 is there were thumb rules for max depth. It gave me one based on # features and one based on number of rows.
    # just using the max of those two for now.  Along with a absolute min of 5 levels
    max_depth_est_1 = np.log10(len(X_test_scaled))
    max_depth_est_2 = np.log2(len(X_train_scaled.columns))
    max_depth_try = round(max(max_depth_est_1, max_depth_est_2, 5))
    

    #Define/select which classifiers to use (For now just use the ones below)  
    lr = LogisticRegression(random_state=42)
    rf = RandomForestClassifier(random_state=42, max_depth=max_depth_try)
    svc = SVC(probability=True, random_state=42)
    gbc = GradientBoostingClassifier(random_state=42)
    abc = AdaBoostClassifier(random_state=42) 
    dt = DecisionTreeClassifier(random_state=42, max_depth=max_depth_try)
    # Create a VotingClassifier with soft voting
    voting_estimators = [
        ('lr', lr), 
        ('rf', rf), 
        ('svc', svc), 
        ('GBC', gbc),
        ('ABC',abc),
        ('DT', dt)]
    
    voting_clf = VotingClassifier(
        estimators = voting_estimators,
        voting = 'soft')
    
    classifier_list = []
    classifier_list.append(lr)
    classifier_list.append(rf)
    classifier_list.append(svc)
    classifier_list.append(gbc)
    classifier_list.append(abc),
    classifier_list.append(dt)
    classifier_list.append(voting_clf)

    #TODO: Add steps to this pipeline for what we want to have
    #define the pipeline
    pipeline = Pipeline(steps=[
        ('classifier', VotingClassifier(
            estimators=voting_estimators,
            voting='soft'))
    ])

    #fit the pipeline
    pipeline.fit(X_train_scaled, y_train)
    
    #run the pipeline
    y_train_predictions = pipeline.predict(X_train_scaled)
    y_test_predictions = pipeline.predict(X_test_scaled)

    # debugging zzz
    # st.write(f"type of y_train: {type(y_train)}")
    # st.write(f"shape of X_train_scaled {np.shape(y_train)}")
    # st.write(f"type of y_train_predictions: {type(y_train_predictions)}")
    # st.write(f"shape of y_train_predictions {np.shape(y_train_predictions)}")
    st.write("\n\n")
    
    voting_class_name = "Voting Classifier Scoring"

    st.write(f"##### **{voting_class_name}**")
    st.write(f"""| Type of Data Set    | Accuracy | Balanced Accuracy |
| -----------------------------------: | :--------------------: | :--------------------: |
| Training | {(accuracy_score(y_train, y_train_predictions)*100):.2f} | {(balanced_accuracy_score(y_train, y_train_predictions)*100):.2f} |
| Test | {(accuracy_score(y_test, y_test_predictions)*100):.2f} | {(balanced_accuracy_score(y_test, y_test_predictions)*100):.2f} |""")
    
    st.write("####\n\n\n")
    
    #get the scores for each classifier Training

    markdown_table = "| **Classifier Name** | **Training Accuracy** | **Training Balanced Accuracy** | **Test Accuracy** | **Test Balanced Accuracy** |\n "
    markdown_table += "| -----------------------------: | :------------------: | :---------------------------: | :------------------: | :---------------------------: |"

    for clf in classifier_list:
        clf.fit(X_train_scaled, y_train)
        y_train_predictions = clf.predict(X_train_scaled)
        y_test_predictions = clf.predict(X_test_scaled)
        train_accuracy = round(accuracy_score(y_train, y_train_predictions)*100,2)
        test_accuracy = round(accuracy_score(y_test, y_test_predictions)*100,2)
        train_bal_accuracy = round(balanced_accuracy_score(y_train, y_train_predictions)*100,2)
        test_bal_accuracy = round(balanced_accuracy_score(y_test, y_test_predictions)*100,2)
        markdown_table += f"\n| {clf.__class__.__name__} | {train_accuracy} | {train_bal_accuracy} | {test_accuracy}| {test_bal_accuracy} | "

    # now display the markdown table
    st.write("#### **Individual Classifier Scoring (Accuracy and Balanced Accuracy)**")
    st.write(markdown_table)

    # Now do the other metrics
    st.write("#### \n\n\n")

    # confusion matrix for train and test
        
    st.write("#### **CONFUSION MATRIX - TRAIN AND TEST** for each model...")
    st.write("\n#### **Key...** ")
    markdown_table_cm_key = f"| **Category** | **Predicted to be False** | **Predicted to be True** |\n "
    markdown_table_cm_key += "| :-----------------: | :------------------------: | :------------------------: | \n"
    markdown_table_cm_key += "| **Actually false** | True Negative (TN)  | False Positive (FP) | \n"
    markdown_table_cm_key += "| **Actually true**  | False Negative (FN) |  True Positive (TP) | "
    st.write(markdown_table_cm_key)
    st.write("#\n\n")

    # Now the confusion matrix for each model
    st.write("""#### CONFUSION MATRIX FOR EACH MODEL
-----------------------------""")
    for clf in classifier_list:
        clf.fit(X_train_scaled, y_train)
        y_train_predictions = clf.predict(X_train_scaled)
        y_test_predictions = clf.predict(X_test_scaled)
        # calculate the confusion matrix for train and test model by model
        cm_train = confusion_matrix(y_train,y_train_predictions)
        TN_train = cm_train[0,0]
        FN_train = cm_train[1,0]
        FP_train = cm_train[0,1]
        TP_train = cm_train[1,1]

        cm_test = confusion_matrix(y_test,y_test_predictions)
        TN_test = cm_test[0,0]
        FN_test = cm_test[1,0]
        FP_test = cm_test[0,1]
        TP_test = cm_test[1,1]

        # Training conf matrix (cm) - write it out
        st.write(f"\n\n##### **Model {clf.__class__.__name__} Training CM**")
        md_table_cm_train = f"| **Category** | **Predicted to be False** | **Predicted to be True** |\n "
        md_table_cm_train += f"| :-----------------: | :------------------------: | :------------------------: | \n"
        md_table_cm_train += f"| Actually false | TN: {TN_train}  | FP: {FP_train} | \n"
        md_table_cm_train += f"| Actually true  | FN: {FN_train}  |  TP: {TP_train} | "
        st.write(md_table_cm_train)
        st.write("#\n\n")

         # Test conf matrix (cm) - write it out
        st.write(f"\n\n##### **Model {clf.__class__.__name__} TEST CM**")
        md_table_cm_test = f"| **Category** | **Predicted to be False** | **Predicted to be True** |\n "
        md_table_cm_test += f"| :-----------------: | :------------------------: | :------------------------: | \n"
        md_table_cm_test += f"| Actually false | TN: {TN_test}  | FP: {FP_test} | \n"
        md_table_cm_test += f"| Actually true  | FN: {FN_test}  | TP: {TP_test} | "
        st.write(md_table_cm_test)
        st.write("#\n\n")




       

    st.write("#\n\n")

    # Now the classification matrix for each model
    st.write("#### Classification Matrix for each model")
    for clf in classifier_list:
        st.write(f"##### Classification Matrix for model {clf.__class__.__name__}")
        clf.fit(X_train_scaled, y_train)
        y_train_predictions = clf.predict(X_train_scaled)
        y_test_predictions = clf.predict(X_test_scaled)
        # classification matrix for train and test
        class_report_train = classification_report(y_train, y_train_predictions)
        class_report_test = classification_report(y_test, y_test_predictions)
        # Now write it out
        st.write(f"##### **Model {clf.__class__.__name__} Training Classification Report**")
        st.text(f"\n{class_report_train}\n\n")
        st.write(f"##### **Model {clf.__class__.__name__} TEST Classification Report**")
        st.text(f"\n{class_report_test}")
        

    st.write("#### \n\n\n")
    st.write("### **Analysis Complete**")
