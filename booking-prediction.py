# Prediction of bookings based on user behavior
# Data Scientist – User Profiling, Hotel Search
#
# Author: Kai Chen
# Date: Apr, 2018

# There are 3 types of data: Bookings, User actions, Example

# Data: Bookings
# - Description: List of sessions, each with: session-related contextual data, and whether at least one booking was made
# - Files:
# 	- case_study_bookings_train.csv: Training sessions for bookings
# 	- case_study_bookings_target.csv: Target sessions to predict bookings
# - Rows: Each row represents a session with session context and the outcome of this session
# - Columns:
# 	- ymd: Date of the session in format 'yyMMdd'
# 	- user_id: Anonymized cookie id of the visitor
# 	- session_id: Anonymized id of the session
# 	- referer_code: Encoded category of the referer to the website
# 	- is_app: If the session was made using the trivago app
# 	- agent_id: Encoded type of the browser
# 	- traffic_type: A categorization of the type of the traffic
# 	- has_booking: 1 if at least one booking was made during the session (excluded from the target set)


# Data: User Actions
# - Description: Sequence of various type of user actions generated during the usage of the website.
# - Files
# 	- case_study_actions_train.csv: Training set of user actions
# 	- case_study_actions_target.csv: User actions in the target sessions
# - Rows: Each row represents one action from/to the user
# - Columns:
# 	- ymd: Date of the action in format 'yyMMdd'
# 	- user_id: Anonymized cookie id of the visitor
# 	- session_id: Anonymized id of the session
# 	- action_id: Type of the action
# 	- reference: Object of the action. - We note that action_ids with big set of reference values (e.g. action id '2116') are typically related to the content (e.g hotels, destinations or keywords); while action_ids with small reference set (e.g. action id '2351') are more related a function of the website (e.g. sorting order, room type, filters, etc.)
# 	- step: The number identifying the action in the session

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier

import lightgbm as lgb

import catboost
from catboost import CatBoostClassifier

# ---
# Define file paths
TRAIN_BOOKING_FILE_PATH = 'data/case_study_bookings_train.csv'    # training sessions for bookings
TARGET_BOOKING_FILE_PATH = 'data/case_study_bookings_target.csv'  # target sessions to predict bookings

TRAIN_ACTION_FILE_PATH = 'data/case_study_actions_train.csv'       # training set of user actions
TARGET_ACTION_FILE_PATH = 'data/case_study_actions_target.csv'     # user actions in the target sessions


# -------------------
# Step 1: read and explore the data
#

# ---
# Define file paths
TRAIN_BOOKING_FILE_PATH = 'data/case_study_bookings_train.csv'    # training sessions for bookings
TARGET_BOOKING_FILE_PATH = 'data/case_study_bookings_target.csv'  # target sessions to predict bookings

TRAIN_ACTION_FILE_PATH = 'data/case_study_actions_train.csv'       # training set of user actions
TARGET_ACTION_FILE_PATH = 'data/case_study_actions_target.csv'     # user actions in the target sessions


train_booking_df = pd.read_csv(TRAIN_BOOKING_FILE_PATH, sep='\t')
train_booking_df['ymd'] = pd.to_datetime(train_booking_df['ymd'].astype('str'))

target_booking_df = pd.read_csv(TARGET_BOOKING_FILE_PATH, sep='\t')
target_booking_df['ymd'] = pd.to_datetime(target_booking_df['ymd'].astype('str'))

train_user_id_list = train_booking_df['user_id'].unique()
train_session_id_list = train_booking_df['session_id'].unique()

target_user_id_list = target_booking_df['user_id'].unique()
target_session_id_list = target_booking_df['session_id'].unique()

train_action_df = pd.read_csv(TRAIN_ACTION_FILE_PATH, sep='\t')
train_action_df['ymd'] = pd.to_datetime(train_action_df['ymd'].astype('str'))

target_action_df = pd.read_csv(TARGET_ACTION_FILE_PATH, sep='\t')
target_action_df['ymd'] = pd.to_datetime(target_action_df['ymd'].astype('str'))


# replace the NAN values by a specific value
NA_ACTION_ID = -10
NA_REFERENCE_ID = -10
NA_STEP = 0

train_user_df = pd.merge(train_booking_df, train_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

train_user_df['action_id'].fillna(NA_ACTION_ID, inplace=True)
train_user_df['reference'].fillna(NA_REFERENCE_ID, inplace=True)
train_user_df['step'].fillna(NA_STEP, inplace=True)

train_user_df['action_id'] = train_user_df['action_id'].astype('int')
train_user_df['reference'] = train_user_df['reference'].astype('int')
train_user_df['step'] = train_user_df['step'].astype('int')

print('-------------------')
print(train_user_df.columns)
print('train user df shape')
print(train_user_df.shape)

target_user_df = pd.merge(target_booking_df, target_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

target_user_df['action_id'].fillna(NA_ACTION_ID, inplace=True)
target_user_df['reference'].fillna(NA_REFERENCE_ID, inplace=True)
target_user_df['step'].fillna(NA_STEP, inplace=True)

target_user_df['action_id'] = target_user_df['action_id'].astype('int')
target_user_df['reference'] = target_user_df['reference'].astype('int')
target_user_df['step'] = target_user_df['step'].astype('int')


print('-------------------')
print(target_user_df.columns)
print('target user df shape')
print(target_user_df.shape)


# ----------
# Step 2. naive approach
feature_columns = ['referer_code', 'is_app', 'agent_id', 'traffic_type', 'action_id', 'reference', 'step']
target_column = ['has_booking']

train_data_df = train_user_df[feature_columns + target_column]

print('-------------------')
print('train data')
print(train_data_df.shape)

train_data_df = train_data_df.reindex(np.random.permutation(train_data_df.index))

train_x = train_data_df[feature_columns]
train_y = train_data_df[target_column].values

# https://stackoverflow.com/questions/31995175/scikit-learn-cross-val-score-too-many-indices-for-array
"""
When we do cross validation in scikit-learn, the process requires an (R,) shape label instead of (R,1). 
Although they are the same thing to some extend, their indexing mechanisms are different. So in your case, just add:
c, r = labels.shape
labels = labels.reshape(c,)
"""
c, r = train_y.shape
train_y = train_y.reshape(c,)

train_sub_x, val_x, train_sub_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

print('-------------------')
print('train')
print(train_x.shape)
print(train_y.shape)


test_x = target_user_df[feature_columns]
# test_y = target_user_df[target_column]

print('-------------------')
print('test')
print(test_x.shape)
# print(test_y.shape)


def timer(start_time=None):
    # fork from https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def train_xgb(X_train, Y_train, hyperparameter_tuning=False, model_path=None, n_jobs=3, folds=3, param_comb=5):
    """
    Train a xgb model

    Reference
    https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    """

    # xgb_clf = XGBClassifier(learning_rate=0.01,
    #                     n_estimators=200,
    #                     objective='binary:logistic',
    #                     silent=True, nthread=nthread)

    xgb_clf = XGBClassifier(nthread=n_jobs, objective='binary:logistic', silent=True,)

    if hyperparameter_tuning:
        print('xgb hyperparameter tuning ...')

        params = {
            #'n_estimators': [100, 200, 400, 500],
            'min_child_weight': [1, 5, 10],
            # 'gamma': [0.5, 1, 1.5, 2, 5],
            'gamma': [0.5, 1, 1.5, 2],
            # 'subsample': [0.6, 0.8, 1.0],
            'subsample': [0.6, 0.8, 1],
            # 'colsample_bytree': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1],
            # 'max_depth': [3, 4, 5]
            'max_depth': [2, 4, 6],
        }

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                           n_jobs=n_jobs,
                                           cv=skf.split(X_train, Y_train),
                                           verbose=3, random_state=42)

        start_time = timer(None)
        random_search.fit(X_train, Y_train)
        timer(start_time)

        print('--------------')
        print('\n all results:')
        print(random_search.cv_results_)

        print('\n best estimator:')
        print(random_search.best_estimator_)

        print('\n best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
        print(random_search.best_score_ * 2 - 1)

        print('\n best xgb hyperparameters:')
        print(random_search.best_params_)

        result_csv_path = 'xgb-random-grid-search-results.csv'
        results = pd.DataFrame(random_search.cv_results_)
        results.to_csv(result_csv_path, index=False)
        print('save xgb random search results to {}'.format(result_csv_path))
        print('--------------')

        xgb_clf = random_search.best_estimator_
    else:
        xgb_clf.fit(train_sub_x, train_sub_y)

    if model_path is None:
        xgb_model_path = 'xgb.model'
        if hyperparameter_tuning:
            xgb_model_path = 'xgb.ht.model'
    else:
        xgb_model_path = model_path
        # xgb_clf.save_model(xgb_model_path)
    joblib.dump(xgb_clf, xgb_model_path)
    print('save xgb model to {}'.format(xgb_model_path))

    return xgb_clf, xgb_model_path


def train_rf(X_train, Y_train, hyperparameter_tuning=False, model_path=None, n_jobs=4, folds=3):
    """
    Train a RF classifier

    Reference
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """
    model = RandomForestClassifier(random_state=42, n_jobs=n_jobs)

    if hyperparameter_tuning:
        # Number of trees in random forest
        #n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        n_estimators = [60, 100, 200, 300]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        #max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth = [4, 6, 8]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        #print(random_grid)

        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid,
                                       n_iter=100, cv=folds, verbose=2, random_state=42, n_jobs=n_jobs)

        rf_random.fit(X_train, X_train)


        print('--------------')
        print('\n all results:')
        print(rf_random.cv_results_)

        print('\n best estimator:')
        print(rf_random.best_estimator_)

        print('\n best rf parameters:')
        print(rf_random.best_params_)

        print('\n best scores:')
        rf_random.best_score_

        result_cv_path = 'rf-random-grid-search-results.csv'
        results = pd.DataFrame(rf_random.cv_results_)
        results.to_csv(result_cv_path, index=False)
        print('\n save rf random search results to {}'.format(result_cv_path))
        print('--------------')

        model = rf_random.best_estimator_
    else:
        model.fit(X_train, Y_train)

    if model_path is None:
        model_path = 'rf.model'
        if hyperparameter_tuning:
            model_path = 'rf.ht.model'


    joblib.dump(model, model_path)
    print('save rf model to {}'.format(model_path))

    return model, model_path


def train_nb(X_train, Y_train, model_path=None):
    # reference https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
    model = GaussianNB()
    model.fit(X_train, Y_train,)

    if model_path is None:
        model_path = 'gnb.model'

    joblib.dump(model, model_path)
    print('save GaussianNB model to {}'.format(model_path))

    return model, model_path


def train_lgbm(X_train, Y_train, categorical_feature=[0, 1, 2, 3, 4, 5, 6],
               model_path=None, n_jobs=3, hyperparameter_tuning=False, num_boost_round=100):
    """
    Train a lightGBM model

    Reference
    https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
    """
    d_train = lgb.Dataset(X_train, label=Y_train,
                          # categorical_feature=['aisle_id', 'department_id']
                          categorical_feature=categorical_feature,
                          )

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'n_jobs': n_jobs,
        #'num_leaves': 31,
        #'learning_rate': 0.05,
        #'feature_fraction': 0.9,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        #'verbose': 0
    }

    gbm = lgb.train(params,
                    d_train,
                    num_boost_round=num_boost_round,
                    categorical_feature=categorical_feature)

    if model_path is None:
        model_path = 'lgbm.model'
        if hyperparameter_tuning:
            model_path = 'lgbm.ht.model'

    # save model to file
    gbm.save_model(model_path)

    # load model to predict
    #print('Load model to predict')
    #bst = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    #y_pred = bst.predict(X_test)

    #TODO: hyperparameter tuning
    # if hyperparameter_tuning:
    #     params = {'boosting_type': 'gbdt',
    #               'max_depth': -1,
    #               'objective': 'binary',
    #               'nthread': 5,  # Updated from nthread
    #               'num_leaves': 64,
    #               'learning_rate': 0.05,
    #               'max_bin': 512,
    #               'subsample_for_bin': 200,
    #               'subsample': 1,
    #               'subsample_freq': 1,
    #               'colsample_bytree': 0.8,
    #               'reg_alpha': 5,
    #               'reg_lambda': 10,
    #               'min_split_gain': 0.5,
    #               'min_child_weight': 1,
    #               'min_child_samples': 5,
    #               'scale_pos_weight': 1,
    #               'num_class': 1,
    #               'metric': 'binary_error'}

    return gbm, model_path


def train_catboost(X_train, Y_train, categorical_feature=[0, 1, 2, 3, 4, 5, 6],
               model_path=None, hyperparameter_tuning=False, num_boost_round=100):

    model = CatBoostClassifier(loss_function='Logloss',
                               iterations=num_boost_round,
                               #learning_rate=1,
                               #depth=2
                               )
    # Fit model
    model.fit(X_train, Y_train, categorical_feature)

    if model_path is None:
        model_path = 'catboost.model'
        if hyperparameter_tuning:
            model_path = 'catboost.ht.model'

    model.save_model(model_path)

    return model, model_path



def predict(model_path, X_test, is_lgbm=False, is_catboost=False, threshold=0.5):
    """
    load the model and predict unseen data
    """
    if is_lgbm:
        # lightgbm
        model = lgb.Booster(model_file=model_path)
    elif is_catboost:
        model = catboost.load_model(model_path)
    else:
        # sklearn
        # xgboost
        model = joblib.load(model_path)


    # y_pred = model.predict_prob(X_test)
    y_pred = model.predict(X_test)

    y_output = []
    for y in y_pred:
        if y > threshold:
            y_output.append(1)
        else:
            y_output.append((0))

    return np.array(y_output)

def predict_blend(X_test, model_paths=['xgb.model', 'rf.model', 'nb.model'], threshold=0.7):
    y_pred = predict(model_paths[0], X_test)

    for i in range(1, len(model_paths)):
        y_pred += predict(model_paths[0], X_test)
    y_pred = y_pred*1.0 / len(model_paths)

    y_output = []
    for y in y_pred:
        if y > threshold:
            y_output.append(1)
        else:
            y_output.append((0))

    return y_output



#model, model_path = train_xgb(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='xgb.model')
#y_pred = predict(model_path, val_x)

#model, model_path = train_lgbm(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='lgbm.model', num_boost_round=100)
#y_pred = predict(model_path, val_x, is_lgbm=True)

model, model_path = train_catboost(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='catboost.model', num_boost_round=100)
y_pred = predict(model_path, val_x, is_catboost=True)

print(y_pred)
print(len(y_pred))
print(type(y_pred))

#model, model_path = train_rf(train_sub_x, train_sub_y, hyperparameter_tuning=True, model_path='rf.ht.model')
#y_pred = predict('rf.ht.model', val_x)

# model, model_path = train_rf(train_sub_x, train_sub_y, hyperparameter_tuning=False)
# model, model_path = train_nb(train_sub_x, train_sub_y, hyperparameter_tuning=False)
# model, model_path = train_lr(train_sub_x, train_sub_y, hyperparameter_tuning=False)


# y_pred = predict_blend(val_x)
# y_pred = predict_blend(train_sub_x)


# --------
# Feature engineering
# category data in machine learning
# Reference:
# https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
# https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009


# ---------
# Evaluate the model
# We expect binary predictions for the target sessions, which will be evaluated by Matthews Correlation Coefficient (MCC)
# using the ground truth dataset on our side.
# The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary (two-class) classifications.
# It takes into account true and false positives and negatives and is generally regarded as a balanced measure
# which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value
# between -1 and +1.
# A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction.
# The statistic is also known as the phi coefficient. [source: Wikipedia]
#
# https://lettier.github.io/posts/2016-08-05-matthews-correlation-coefficient.html


y_true = val_y
# y_true = train_sub_y

print(y_true)
print(len(y_true))
print(type(y_true))

score = matthews_corrcoef(y_true, y_pred)
print('matthews corrcoef score')
print(score)

accuracy = accuracy_score(y_true, y_pred)
print('accuracy: {}'.format(accuracy))
print(classification_report(y_true, y_pred))


