import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, accuracy_score


def calc_metrics(X_train, X_test, y_train, y_test, model, random_seed_num):
    '''
    Function calculates various metrics for classification models.
    This assumes the X train & X test are already scaled or coded.
    '''
    
    print("In calc_metrics, model class is: ",model.__class__)
    try:
        model.fit(X_train,y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    except:
        print("in calc_metrics, issue either fitting or using model to predict")
        return False
    try:
        accuracy_score_model_train = accuracy_score(y_train, y_pred_train)
        accuracy_score_model_test = accuracy_score(y_test, y_pred_test)
        print(f"\nAccuarcy Score for training data is: {(accuracy_score_model_train*100):.2f}")
        print(  f"Accuarcy Score for test data is:     {(accuracy_score_model_test*100):.2f}")
    except:
        print("could not calculate accuracy score for train or test (at least one of train or test)")
        accuracy_score_model_train = pd.NA
        accuracy_score_model_test = pd.NA
    try:
        # confusion matrix for train and test
        conf_matrix_train = confusion_matrix(y_train,y_pred_train)
        conf_matrix_test = confusion_matrix(y_test,y_pred_test)
        print("\n\nCONFUSION MATRIX - TRAIN AND TEST...")
        print("\nKey...")
        print("-------------------------------------------------------------------------")
        print("                |   Predicted to be False     |  Predicted to be True   |")
        print("Actually false  |     True Negative  (TN)     |     False Positive (FP) |")
        print("Actually true   |     False Negative (FN)     |      True Positive (TP) |")
        print("-------------------------------------------------------------------------")
        print(f"\n\nConfusion Matrix using Training Data:\n{conf_matrix_train}\n\n")
        print("\nConfusion Matrix using Test Data:\n", conf_matrix_test)
    except:
        print("could not calculate confusion matrix for either train or test ")
        conf_matrix_train = pd.NA
        conf_matrix_test = pd.NA
    try:
        # now run the classification_report for training and test
        
        class_report_train = classification_report(y_train, y_pred_train)
        class_report_test = classification_report(y_test, y_pred_test)
        print("\n\nClassification Report for Train:\n", class_report_train)
        print("\nClassification Report for Test:\n", class_report_test)

    except:
        print("Error running classification report on either train or test")
        class_report_train = pd.NA
        class_report_test = pd.NA
    try:
        # Calculate the balanced accuracy score
        balanced_acc_score_train = balanced_accuracy_score(y_train, y_pred_train)
        balanced_acc_score_test = balanced_accuracy_score(y_test, y_pred_test)
        print(f"Balanced Accuracy Score - Train: {balanced_acc_score_train*100:.2f}")
        print(f"Balanced Accuracy Score - Test:  {balanced_acc_score_test*100:.2f}")
    except:
        balanced_acc_score_train = pd.NA
        balanced_acc_score_test = pd.NA
    try:
        # Predict values with probabilities
        y_pred_prob_train = model.predict_proba(X_train)
        y_pred_prob_test = model.predict_proba(X_test)
        y_pred_proba_train_1 = y_pred_prob_train[:,1]
        y_pred_proba_test_1 = y_pred_prob_test[:,1]
        y_pred_prob_train_list = [y_pred_prob_train[i,1] for i in range(1,len(y_pred_prob_train))]
        y_pred_prob_test_list = [y_pred_prob_test[i,1] for i in range(1,len(y_pred_prob_test))]
        print("First 5 values of y probability for train...")
        print(y_pred_prob_train_list[:5])
        print("First 5 values of y probability for test...")
        print(y_pred_prob_test_list[:5])

    except:
        print("\nCould not calculate predicted probabilities")
        y_pred_prob_train_list = pd.NA
        y_pred_prob_test_list = pd.NA
        y_pred_proba_train_1 = pd.NA
        y_pred_proba_test_1 = pd.NA
    try:
        # ROC AUC Score
        roc_auc_train = roc_auc_score(y_train, y_pred_proba_train_1)
        roc_auc_test = roc_auc_score(y_test, y_pred_proba_test_1)
        print("\n\n")
        print(f"ROC AUC Score for Training: {roc_auc_train*100:.2f}")
        print(f"ROC AUC Score for Test:     {roc_auc_test*100:.2f}")
    except:
        print("\n\nCould not calculate ROC AUC scores")
        roc_auc_train = pd.NA
        roc_auc_test = pd.NA

    print("\n\nMetric Report complete")
    return {'model_type':model.__class__,
            'accuracy_score_model_train': accuracy_score_model_train,
            'accuracy_score_model_test': accuracy_score_model_test,
            'conf_matrix_train':conf_matrix_train,
            'conf_matrix_test': conf_matrix_test,
            'Classification_Report_Train': class_report_train,
            'Classification_Report_Test': class_report_test,
            'Balanced_Accuracy_Score_Train': balanced_acc_score_train,
            'Balanced_Accuracy_Score_Test': balanced_acc_score_test,
            'y_pred_prob_train': y_pred_prob_train_list,
            'y_pred_prob_test': y_pred_prob_test_list,
            'ROC_Score_Train': roc_auc_train,
            'ROC_Score_Test': roc_auc_test,
            }