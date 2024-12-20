
import sys
import numpy as np
import pandas as pd
import sklearn as sk
import scipy as sc
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import optparse
import random
random.seed(123)
np.random.seed(123)

def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run decision tree method')

    parser.add_option('-r', '--train_filename', type='string', help='path to' +\
        ' train arff file')
    parser.add_option('-e', '--test_filename', type='string', help='path to' +\
        ' test arff file')
    parser.add_option('-a', '--alpha', type='float', default=0.1, help='alpha (optional)')
    parser.add_option('-l', '--lmda', type='float', default=0.1, help='lmda (optional)')

    (opts, args) = parser.parse_args()

    mandatories = ['train_filename', 'test_filename',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def parse_csv(filename):
    #y_pred = prediction(X, coef)
    f = open(filename,'r')
    X = []
    y = []
    line_one = f.readline()
    for line in f:
        #import pdb; pdb.set_trace()
        tokens = line.strip().split(",")
        X.append([float(x) for x in tokens[0:-1]])
        y_val = int(int(tokens[-1])==1)
        y.append(y_val)
    X = np.array(X)
    y = np.array(y)
    return X,y

def log_col(data, param):
    y = data[param]
    X = data.drop(columns=[param])
    
    log = LogisticRegression()
    true_vals = list(data[param])
    ret_accs = []
    for i in range(int(len(X.columns) / 2)):
        col_acc = 1.0
        ret_col = ""
        for c in X.columns:
            curr_cols = X.drop(columns=c)
            curr_data = pd.get_dummies(curr_cols, dtype=float)
            log = LogisticRegression(max_iter=500)
            X_train, X_test, y_train, y_test = train_test_split(curr_data, y)
            log.fit(X_train, y_train)
            score = log.score(X_test, y_test)
            y_pred = log.predict(X_test)
            b_score = sk.metrics.balanced_accuracy_score(y_test, y_pred).item()
            if b_score < col_acc: 
                col_acc = b_score
                ret_col = c
        ret_accs.append([ret_col, col_acc])
        X = X.drop(columns=ret_col)
    return ret_accs
def tree_col(data, param):
    y = data[param]
    X = data.drop(columns=[param])
    
    tree = DecisionTreeClassifier()
    true_vals = list(data[param])
    ret_accs = []
    for i in range(int(len(X.columns) / 2)):
        col_acc = 1.0
        ret_col = ""
        for c in X.columns:
            curr_cols = X.drop(columns=c)
            curr_data = pd.get_dummies(curr_cols, dtype=float)
            X_train, X_test, y_train, y_test = train_test_split(curr_data, y)
            tree.fit(X_train, y_train)
            score = tree.score(X_test, y_test)
            y_pred = tree.predict(X_test)
            b_score = sk.metrics.balanced_accuracy_score(y_test, y_pred).item()
            if b_score < col_acc: 
                col_acc = b_score
                ret_col = c
        ret_accs.append([ret_col, col_acc])
        X = X.drop(columns=ret_col)
    return ret_accs

def run_tune_test(learner, params, X, y):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    parameters = params
    grid_scores = []
    grid = GridSearchCV(learner, parameters)
    best_score = 0
    best_param = ""
    y_preds = []
    pre_score = 0
    rec_score = 0
    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        grid.fit(X[train_index], y[train_index])
        score = grid.score(X[train_index], y[train_index])
        '''
        print(f"Fold {i + 1}:")
        print("Params: ", grid.best_params_)
        print(f"Training Score: {score}")
        '''
        grid_preds = grid.predict(X[test_index])
        test_score = sk.metrics.balanced_accuracy_score(grid_preds, y[test_index]).item()
        grid_scores.append(test_score)
        if test_score > best_score: 
            best_score = test_score
            best_param = grid.best_params_
            y_preds = grid_preds
            pre_score = sk.metrics.average_precision_score(y[test_index], y_preds)
            rec_score = sk.metrics.recall_score(y[test_index], y_preds)
    f1_score =  (2 * pre_score * rec_score) / (pre_score + rec_score)
    print("f1 score: ", f1_score)
    return grid_scores, (best_score, best_param)

def main():
    #use the feature vectorizer! Fit train, apply train, fit test, apply test. 
    
    opts = parse_args()
    vectorizer = CountVectorizer()
    train_data = pd.read_csv(opts.train_filename)
    test_data = pd.read_csv(opts.test_filename)
    #all_data = pd.concat([train_data, test_data])
    all_data = train_data
    log_params = {"penalty": ['l1', 'l2', 'elasticnet', None], "C": [1, 10, 100, 1000], "max_iter": [500]}
    tree_params = {"criterion": ['gini', 'entropy', 'log_loss'], "max_features": [None, 'sqrt', 'log2', 0.01, 0.1, 0.5, 1.0]}
    all_data['austim'] = all_data['austim'].replace('yes', 1.0) 
    all_data['austim'] = all_data['austim'].replace('no', 0.0)

    all_data['Class/ASD'] = all_data['Class/ASD'].replace(1, 1.0)
    all_data['Class/ASD'] = all_data['Class/ASD'].replace(0, 0.0)
    all_data = all_data.drop(columns=['age_desc', 'ID'])
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    all_two = pd.get_dummies(all_data, dtype=float)
    
    na_vals = all_two.isna()
    min_max_scaler = MinMaxScaler()
    
    y_data = all_two['Class/ASD']
    x_data = all_two.drop(columns=['Class/ASD'])

    
    imp_mean.fit(x_data)
    final_x = imp_mean.transform(x_data)

    
    final_x = min_max_scaler.fit_transform(final_x)
    log = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(final_x, y_data)
    clf = DecisionTreeClassifier()
    #clf.fit(X_train, y_train)
    #dec_pred = clf.predict(X_test)
    #print(sk.metrics.balanced_accuracy_score(dec_pred, y_test))
    #log.fit(X_train, y_train)
    #best_col = []
    #curr_data = all_data.drop(columns=['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score'])
    curr_data = all_data
    '''
    log_results, log_par = run_tune_test(log, log_params, final_x, y_data)
    tree_results, tree_par = run_tune_test(clf, tree_params, final_x, y_data)
    '''
    #print("Log results: ", log_results)
    #print("Log parameters and score", log_par)
    #print("Tree results: ", tree_results)
    #print("Tree parameters and score", tree_par)
    log_best_col = (log_col(curr_data, 'Class/ASD'))
    tree_best_col = (tree_col(curr_data, 'Class/ASD'))
    print("Log columns: ", log_best_col)
    print("Tree columns: ", tree_best_col)
    #log_preds = log.predict(X_test)
    #print(log.score(X_test, y_test)) #sk.metrics.balanced_accuracy_score(log_preds, y_test)
    
    #print(b_score)
    #print(sorted_dict)  
    '''
    Do ablation report(?): instead of testing one feature at a time, run models with individual features excluded. Original model: ABCD, Round 1: ABC, ABD, ACD, BCD then, 
    if B is the best feature it goes down to two features: AC, CD, AD. Ends with half features.
    Prescision is true positive rate (tp/(tp+fp)), recall is tp/(fn+tp)
    '''

main()
