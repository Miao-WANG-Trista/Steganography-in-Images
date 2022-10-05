import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import argparse
import os
import numpy as np
from catboost import CatBoostClassifier, Pool
import sys
sys.path.insert(1,'./')
from train_module.tools.kaggle_tools import wauc
from skopt import BayesSearchCV

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def split(df, subset):
    """
    process the dataframe to generate 'cate' and 'label' as target variables, identify categorical features, standardize numerical features, identify target variable
    and do train-validation test split

    Return
    x_train, x_test, y_train, y_test
    """
    if subset == '3Algorithms':
        cate_dict = {'ver': 0, 'POD': 1, 'ARD': 2, 'ERD': 3}
    elif subset == 'for_nsf5':
        cate_dict = {'ver': 0, 'sf5': 1}
    df['cate'] = df['NAME'].apply(lambda x: cate_dict[x[-7:-4]])
    df['label'] = df['cate'].apply(lambda x: 0 if x == 0 else 1)

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['QF'] = df['QF'].astype('int').astype('str') # 'QF' should be converted into str cuz it's categorical feature rather than numerical
    without = df.drop(columns=['label','NAME','cate']) # remove columns not for training
    scaler = StandardScaler() # apply standardization on numerical columns to avoid the influence of scale
    num_cols = without.columns[without.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
    without[num_cols] = scaler.fit_transform(without[num_cols])
    x_train, x_test, y_train, y_test = train_test_split(without, df['label'], test_size=0.2, random_state=2)
    # here I put 'label' as target variable to do binary classification, otherwise for multi-class classification, put 'cate' here
    return x_train, x_test, y_train, y_test


def print_best_score(gsearch, param_test):

    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")

    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--zoo-file', type=str, default='models_predictions/LB/probabilities_zoo_lb.csv', help='path to zoo file')
    arg('--weights_dir', type=str, default='weights/catboost/', help='path to catboost train_module dir')
    arg('--n-splits', type=int, default=3 , help='num CV splits')
    arg('--subset', type=str, default='3Algorithms', help='the folder for three algorithms or nsf5?')


    args = parser.parse_args()

    if not os.path.exists(args.weights_dir):  # create a directory if path to store catboost weights doesn't exist
        os.makedirs(args.weights_dir)
    df = pd.read_csv(args.zoo_file,index_col=0)
    x_train, x_test, y_train, y_test = split(df,subset=args.subset)

    # Catboost training
    # indicating categorical features

    categorical_features_indices = np.where(x_train.dtypes != float)[0]
    train_pool = Pool(x_train, y_train, cat_features=categorical_features_indices)

    # GridSearch for hyperparameter tuning
    params_cat = {'depth': [2,4,6,8,10],
              'iterations':[100,300,500],
              'learning_rate':[0.01,0.1,0.5],
              'l2_leaf_reg':[1,10,50]}
    # params_cat = {'depth': [12],
    #           'iterations':[100],
    #           'learning_rate':[0.01,0.01],
    #           'l2_leaf_reg':[50]}
    cat = CatBoostClassifier()
    CV_cat = GridSearchCV(cat,param_grid=params_cat,scoring='f1',cv=args.n_splits)
    CV_cat.fit(x_train,y_train,cat_features=categorical_features_indices)

    print_best_score(CV_cat,params_cat)


    CV_cat.best_estimator_.save_model(args.weights_dir + 'best_catboost.cmb')
    print('model weights saved!')

    y_pred_cat = CV_cat.best_estimator_.predict(x_test)

    # Feature importance
    print(CV_cat.best_estimator_.get_feature_importance(prettified=True))
    # Accuracy score
    print("Accuracy for Catboost on CV data: ", accuracy_score(y_test, y_pred_cat))

    # Classification report

    print(classification_report(y_test, y_pred_cat, labels=np.unique(df['label'])))

    # Confusion matrix
    labels_name = ['Innocent', 'Malicious']
    cm = confusion_matrix(y_test, y_pred_cat, labels=np.unique(df['label']))
    print('consufion matrix is', cm)

    sns.set()
    f, ax = plt.subplots()
    svm = sns.heatmap(cm, annot=True, ax=ax, fmt='.20g')

    ax.set_title('confusion matrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    ax.set_xticklabels(labels_name)
    ax.set_yticklabels(labels_name)
    figure = svm.get_figure()
    figure.savefig('catboost_results.png', dpi=400)

if __name__ == "__main__":
    main()

