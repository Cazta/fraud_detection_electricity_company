# import libraries

# import necessary libraries

# for display dataframe
from IPython.display import display 

import pandas as pd
import numpy as np
import missingno as msno 
import seaborn as sns
import matplotlib.pyplot as plt 

#sklearn

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, balanced_accuracy_score


# load, get information on dataset and display descriptive summary
def get_data_summary(data_path = None, data= None, desc_sm = False, no_unq = False, *args, **kwargs): 

    if (data is None) and (data_path is None):
        raise ValueError('''Either enter a data path or a dataset (dataframe)
                    
                        'data' : a dataset (dataframe)
                        'datapath' : a data path used to load a csv data file
                    
                        ''') 

    elif (data is None) and (data_path is not None):
        data = pd.read_csv( data_path) 
    else:
        data = data


    print (f"Dataset shape: {data.shape}") 

    print('_____'*10)

    print(f''' 
    Number of observations : {data.shape[0]}
    Number of features : {data.shape[1]}
        ''')

    print('_____'*10)

    print ("Dataset sample: ") 
    print('_____'*10)

    display(data.sample(n=20, random_state=1))

    print('_____'*10)

    if desc_sm:
        print ("Dataset descriptive summary: ") 
        print('_____'*10)

        display(data.describe().T.style.format('{:.2f}'))

    print('_____'*10)

    if no_unq:
        print ("Unique values/classes for dataset features: ") 
        print('_____'*10)

        display(data.nunique())


    return data 


# write function displaying our model metrics
def our_metrics(y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
        print('Model Metrics and Normalized Confusion Matrix')
        print("_____________________")
        print("_____________________")
    else:
        print('Model Metrics and Confusion Matrix without Normalization')
        print("_____________________")
        print("_____________________")
        sns.heatmap(cm, cmap="YlGnBu", annot=True);
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("_____________________")
    print('F1-score:', round(f1_score(y_true, y_pred), 4))
    print("_____________________")
    print('Fbeta_score with beta=1.5:', round(fbeta_score(y_true, y_pred, beta=1.5), 4)) 
    print("_____________________")
    print('Fbeta_score with beta=2:', round(fbeta_score(y_true, y_pred, beta=2), 4)) 
    print("_____________________")
    print('Fbeta_score with beta=3:', round(fbeta_score(y_true, y_pred, beta=3), 4)) 
    print("_____________________")
    print('Recall', round(recall_score(y_true, y_pred), 4))
    print("_____________________")
    print('Specificity', round(recall_score(y_true, y_pred, pos_label=0), 4))

    
# make the Fbeta scorers needed for the grid search
def get_f15():
    f15_scorer = make_scorer(fbeta_score, beta=1.5)
    return f15_scorer

def get_f2():
    ftwo_scorer = make_scorer(fbeta_score, beta=2)
    return ftwo_scorer

def get_f3():
    fthree_scorer = make_scorer(fbeta_score, beta=3)
    return fthree_scorer

# evaluation metrics : confusion matrix,, accuracy, balance accuracy, classification report
def eval_metrics(y_test, y_pred): 
    """
    Summary:
        Function to calculate the accuracy and balanced accuracy score for imbalanced data, get the confusion 
        matrix as well as the classification report of the ML 
        model based on the predictions and true target values for the test set.

    Args:
        y_test (numpy.ndarray): test target data
        y_pred (numpy.ndarray): predictions based on test data
    """    
    
    print("-----"*15)
    print(f'''Confusion Matrix: 
    {confusion_matrix(y_test, y_pred)} ''') 
    
    print("-----"*15)
    print (f''' Accuracy : 
    {(accuracy_score(y_test, y_pred).round(2)) * 100} ''')

    print("-----"*15)
    print (f''' Balanced Accuracy : 
    {(balanced_accuracy_score(y_test, y_pred).round(2)) * 100} ''')
    
    print("-----"*15)
    print(f'''Report :  
    {classification_report(y_test, y_pred)} ''') 


# # eval scoring metrics : recall, precision, f1_score, roc_auc_score, fpr, tpr
# def evaluate_model(predictions, probs, train_predictions, train_probs):
#     """Compare machine learning model to baseline performance.
#     Computes statistics and shows ROC curve."""
    
#     baseline = {}
    
#     baseline['recall'] = recall_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['precision'] = precision_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['f1_score'] = f1_score(y_test, [1 for _ in range(len(y_test))])
#     baseline['roc'] = 0.5
    
#     results = {}
    
#     results['recall'] = recall_score(y_test, predictions)
#     results['precision'] = precision_score(y_test, predictions)
#     results['f1_score'] = f1_score(y_test, predictions)
#     results['roc'] = roc_auc_score(y_test, probs)
    
#     # train_results = {}
#     # train_results['recall'] = recall_score(y_test, train_predictions)
#     # train_results['precision'] = precision_score(y_test, train_predictions)
#     # train_results['f1_score'] = f1_score(y_test, predictions)
#     # train_results['roc'] = roc_auc_score(y_test, train_probs)
    
#     for metric in ['recall', 'precision', 'f1_score', 'roc']:
#         #print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} Train: {round(train_results[metric], 2)}')
#         print(f'{metric.capitalize()} Baseline: {round(baseline[metric], 2)} Test: {round(results[metric], 2)} ')

#     # Calculate false positive rates and true positive rates
#     base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])
#     model_fpr, model_tpr, _ = roc_curve(y_test, probs)

#     plt.figure(figsize = (8, 6))
#     plt.rcParams['font.size'] = 16
    
#     # Plot both curves
#     plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
#     plt.plot(model_fpr, model_tpr, 'r', label = 'model')
#     plt.legend()
#     plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')

if (__name__ == "__main__"):
    print(get_f15())
    print(get_f2())
    print(get_f3())



import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 


def describe_my_model(fitted_model):
  """
  returns est. model intercept, coefficient and set model hyperparamenters.
  required arguments: fitted model
  """
  # return est. model intercept as df
  interc_dict = {' ' : fitted_model.intercept_[0]}
  df_model_intercept = pd.DataFrame.from_dict(interc_dict, orient='index', columns=['est. model intercept'])

  # return est. feature coefficients as df:
  coef_list = []
  coef_dict = {}
  for coef in fitted_model.coef_:
    coef_list = coef.tolist()
  keys = range(len(coef_list))
  values = coef_list
  for i in keys:
    coef_dict[i] = values[i]
  df_model_coef = pd.DataFrame.from_dict(coef_dict, orient="index", columns=['est. feature coefficients'])

  # return used hyperparameters as df:  
  df_model_params = pd.DataFrame.from_dict(fitted_model.get_params(), orient="index", columns=['set hyperparameters'])

  # display all df:
  display(df_model_intercept, df_model_coef, df_model_params)




def class_metrics_var_threshold(fitted_model, X_test, y_test, threshold=0.5):
    """
    function to compute confusion matrix and classification metrics based on passed threshold.
    required arguments: fitted model, X_test, y_test, threshold (float, default >= 0.5)
    """
    # predicted probabilities based on fitted model
    proba = fitted_model.predict_proba(X_test)
    
    # predicted y based on passed threshold
    y_pred = [int(i>=threshold) for i in proba[:,1]]

    # random line for nicer output :)
    print('____________________')

    # confusion matrix of actual y and predicted y
    print ('Confusion Matrix :')
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print('____________________')

    # accuracy score of confusion matrix
    print ('Accuracy Score :',accuracy_score(y_test, y_pred) )
    print('____________________')

    # classification report of confusion matrix
    print ('Report : ')
    print (classification_report(y_test, y_pred) )
    print('____________________')
    return proba