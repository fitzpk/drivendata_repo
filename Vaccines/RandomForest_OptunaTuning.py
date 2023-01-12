# -------------------------------------
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import optuna
import patsy
import re


# -------------------------------------
# IMPORT DATA

dir_path = os.path.dirname(os.path.realpath(__file__))

train = pd.read_csv(dir_path+'/training_set_features.csv')
train_labels = pd.read_csv(dir_path+'/training_set_labels.csv')
test = pd.read_csv(dir_path+'/test_set_features.csv')
sub_format = pd.read_csv(dir_path+'/submission_format.csv')



# -------------------------------------
# IMPUTATION STRATEGIES

def simple_imputation(train_set,test_set):

    train_nums = train_set.select_dtypes('number')
    test_nums = test_set.select_dtypes('number')

    train_nums = train_nums.fillna(value=-1)
    test_nums = test_nums.fillna(value=-1)

    # Now simply fill with 'None' for the categorical features
    train_cats = train.select_dtypes('object')
    test_cats = test.select_dtypes('object')

    train_cats = train_cats.fillna(value='None')
    test_cats = test_cats.fillna(value='None')

    train_df = pd.concat([train_nums,train_cats],axis=1)
    test_df = pd.concat([test_nums,test_cats],axis=1)

    return train_df,test_df


def mode_imputation(train_set,test_set):
    numeric_cols = list(train_set.select_dtypes('number').columns.values)
    other_cols = list(train_set.select_dtypes('object').columns.values)

    num_imputer = SimpleImputer(strategy='most_frequent').fit(train_set[numeric_cols])
    train_num = pd.DataFrame(num_imputer.transform(train_set[numeric_cols]), columns = train_set[numeric_cols].columns)
    test_num = pd.DataFrame(num_imputer.transform(test_set[numeric_cols]), columns = test_set[numeric_cols].columns)

    cat_imputer = SimpleImputer(strategy='most_frequent').fit(train_set[other_cols])
    train_cat = pd.DataFrame(cat_imputer.transform(train_set[other_cols]), columns = train_set[other_cols].columns)
    test_cat = pd.DataFrame(cat_imputer.transform(test_set[other_cols]), columns = test_set[other_cols].columns)

    train_df = pd.concat([train_num,train_cat],axis=1)
    test_df = pd.concat([test_num,test_cat],axis=1)

    return train_df,test_df


def mean_none_imputation(train_set,test_set):
    numeric_cols = list(train_set.select_dtypes('number').columns.values)
    other_cols = list(train_set.select_dtypes('object').columns.values)

    num_imputer = SimpleImputer(strategy='mean').fit(train_set[numeric_cols])
    train_nums = pd.DataFrame(num_imputer.transform(train_set[numeric_cols]), columns = train_set[numeric_cols].columns)
    test_nums = pd.DataFrame(num_imputer.transform(test_set[numeric_cols]), columns = test_set[numeric_cols].columns)

    train_nums = round(train_nums,0)
    test_nums = round(test_nums,0)

    # Now simply fill with 'None' for the categorical features
    train_cats = train_set.select_dtypes('object')
    test_cats = test_set.select_dtypes('object')

    train_cats = train_cats.fillna(value='None')
    test_cats = test_cats.fillna(value='None')

    train_df = pd.concat([train_nums,train_cats],axis=1)
    test_df = pd.concat([train_nums,test_cats],axis=1)

    return train_df,test_df


train,test = simple_imputation(train,test)
#train,test = mean_none_imputation(train,test)
#train,test = mode_imputation(train,test)



# -------------------------------------
# ORDINAL ENCODING

ordinals = ['age_group','education','employment_status','income_poverty']
ord_encoder = OrdinalEncoder()
data = pd.concat([train,test],axis=0)
data = data.reset_index()
data = data.drop(['index'],axis=1)
ordinals_trns = pd.DataFrame(ord_encoder.fit_transform(data[ordinals]),columns=data[ordinals].columns)
data = pd.concat([data.drop(ordinals,axis=1),ordinals_trns],axis=1)


# -------------------------------------
# DUMMY ENCODE CATEGORICAL VARIABLES THAT WEREN'T USED IN INTERACTION MATRIX ABOVE

# Loop through object columns and transform to dummy variable
collector = pd.DataFrame()
for col in data.select_dtypes('object'):
    col_dummies = pd.get_dummies(data[col], drop_first=True, prefix=col, prefix_sep='_')
    collector = pd.concat([collector, col_dummies], axis=1)

# Combine encoded object data with numeric data
data = pd.concat([data.select_dtypes(['number']),collector],axis=1)

data.columns = data.columns.str.replace('<', '')
data.columns = data.columns.str.replace('>', '')
data.columns = data.columns.str.replace('[', '')
data.columns = data.columns.str.replace(']', '')
data.columns = data.columns.str.replace('+', '')
data.columns = data.columns.str.replace('.', '')
data.columns = data.columns.str.replace(':', '')
data.columns = data.columns.str.replace(',', '')


# Separate back into training and testing
train = data[data['respondent_id'] <= 26706]
test = data[data['respondent_id'] > 26706]




# -------------------------------------
# XY DATA SPLIT

X_train = train.drop(['respondent_id'],axis=1)
y_train_h1n1 = train_labels['h1n1_vaccine']
y_train_seas = train_labels['seasonal_vaccine']
X_test = test.drop(['respondent_id'],axis=1)



# -------------------------------------
# TUNING AND FITTING

def objective_h1n1(trial):
    
    # Specify a search space using distributions across plausible values of hyperparameters.
    param_grid = {
        "verbose": 1,

        'n_estimators': trial.suggest_int('n_estimators', low=4, high=300),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",low=4, high=2**15, log=True),
        "max_features": trial.suggest_loguniform('max_features', low=0.1, high=1),

        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 100),
        'min_impurity_decrease': trial.suggest_int('min_impurity_decrease', 0.00000001, 0.5),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        #"max_samples": trial.suggest_float('max_samples', 0, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 11),
        
    }

    rfc = sklearn.ensemble.RandomForestClassifier(**param_grid)

    print("Trial Number:",trial.number)
    print("Trial Params:",trial.params)
    
    skf = StratifiedKFold(n_splits=10)

    cv_mean = cross_val_score(rfc, X_train, y_train_h1n1, cv=skf, scoring='roc_auc').mean()
    print(cv_mean,"\n")
    
    return cv_mean

# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
optuna.logging.set_verbosity(optuna.logging.WARNING)

study_h1n1 = optuna.create_study(direction='maximize')
study_h1n1.optimize(objective_h1n1, n_trials=1000)

print("Number of finished trials: ", len(study_h1n1.trials))
print("Best trial:")
trial_h1n1 = study_h1n1.best_trial

print("  Value: {}".format(trial_h1n1.value))
print("  Params: ")
for key, value in trial_h1n1.params.items():
    print("    {}: {}".format(key, value))





def objective_seas(trial):
    
    # Specify a search space using distributions across plausible values of hyperparameters.
    param_grid = {
        "verbose": 1,

        'n_estimators': trial.suggest_int('n_estimators', low=4, high=300),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes",low=4, high=2**15, log=True),
        "max_features": trial.suggest_loguniform('max_features', low=0.1, high=1),

        'min_samples_split': trial.suggest_int('min_samples_split', 2, 100),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 100),
        'min_impurity_decrease': trial.suggest_int('min_impurity_decrease', 0.00000001, 0.5),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        #"max_samples": trial.suggest_float('max_samples', 0, 1),
        'max_depth': trial.suggest_int('max_depth', 1, 11),
    }

    rfc = sklearn.ensemble.RandomForestClassifier(**param_grid)

    print("Trial Number:",trial.number)
    print("Trial Params:",trial.params)
    
    skf = StratifiedKFold(n_splits=10)

    cv_mean = cross_val_score(rfc, X_train, y_train_seas, n_jobs=-1, cv=skf, scoring='roc_auc').mean()
    print(cv_mean,"\n")
    
    return cv_mean

# Suppress information only outputs - otherwise optuna is 
# quite verbose, which can be nice, but takes up a lot of space
optuna.logging.set_verbosity(optuna.logging.WARNING)

study_seas = optuna.create_study(direction='maximize')
study_seas.optimize(objective_seas, n_trials=1000)

print("Number of finished trials: ", len(study_seas.trials))
print("Best trial:")
trial_seas = study_seas.best_trial

print("  Value: {}".format(trial_seas.value))
print("  Params: ")
for key, value in trial_seas.params.items():
    print("    {}: {}".format(key, value))



# -------------------------------------
# CREATE SUBMISSION FILE 

final_h1n1 = sklearn.ensemble.RandomForestClassifier(****trial_h1n1.params)
final_h1n1.fit(X_train, y_train_h1n1)

# Generate predictions
y_pred_h1n1 = final_h1n1.predict_proba(X_test)


final_seas = sklearn.ensemble.RandomForestClassifier(**trial_seas.params)
final_seas.fit(X_train, y_train_seas)

# Generate predictions
y_pred_seas = final_seas.predict_proba(X_test)


# Place predictions in dataframe
submission = pd.DataFrame({'respondent_id':sub_format['respondent_id'],
              'h1n1_vaccine':list(y_pred_h1n1[:,1]),
              'seasonal_vaccine':list(y_pred_seas[:,1])})

submission.to_csv(dir_path+'/submission_RandomForest_1000Trials.csv',index=False)



