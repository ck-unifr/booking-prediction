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

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from xgboost import XGBClassifier

# ---
# Define file paths
TRAIN_BOOKING_FILE_PATH = 'data/case_study_bookings_train.csv'    # training sessions for bookings
TARGET_BOOKING_FILE_PATH = 'data/case_study_bookings_target.csv'  # target sessions to predict bookings

TRAIN_ACTION_FILE_PATH = 'data/case_study_actions_train.csv'       # training set of user actions
TARGET_ACTION_FILE_PATH = 'data/case_study_actions_target.csv'     # user actions in the target sessions


train_booking_df = pd.read_csv(TRAIN_BOOKING_FILE_PATH, sep='\t')
train_booking_df['ymd'] = pd.to_datetime(train_booking_df['ymd'].astype('str'))

print('train booking')
print(train_booking_df.columns)
print(train_booking_df.head(5))

target_booking_df = pd.read_csv(TARGET_BOOKING_FILE_PATH, sep='\t')
target_booking_df['ymd'] = pd.to_datetime(target_booking_df['ymd'].astype('str'))

print('target booking')
print(target_booking_df.columns)
print(target_booking_df.head(10))

train_user_id_list = train_booking_df['user_id'].unique()
train_session_id_list = train_booking_df['session_id'].unique()

print('---------------')
print('number of users (train booking data): {}'.format(len(train_user_id_list)))
print('number of sessions (tarin booking data): {}'.format(len(train_session_id_list)))
print('dataframe size (train booking data)')
print(train_booking_df.shape)
print('---------------')

target_user_id_list = target_booking_df['user_id'].unique()
target_session_id_list = target_booking_df['session_id'].unique()

print('---------------')
print('number of users (target booking data): {}'.format(len(target_user_id_list)))
print('number of sessions (target booking data): {}'.format(len(target_session_id_list)))
print('dataframe size (target booking data)')
print(target_booking_df.shape)
print('---------------')


# print('---------------')
# print(train_booking_df[train_booking_df['user_id'] == 388309106223940])
# print('---------------')
# print(train_booking_df[train_booking_df['user_id'] == 452426828488840])


train_action_df = pd.read_csv(TRAIN_ACTION_FILE_PATH, sep='\t')
train_action_df['ymd'] = pd.to_datetime(train_action_df['ymd'].astype('str'))

print('train action')
print(train_action_df.columns)
print(train_action_df.head(5))

train_user_id_action_list = train_action_df['user_id'].unique()
train_session_id_action_list = train_action_df['session_id'].unique()

print('---------------')
print('number of users (train action data): {}'.format(len(train_user_id_action_list)))
print('number of sessions (train action data): {}'.format(len(train_session_id_action_list)))
print('dataframe size (train booking data)')
print(train_action_df.shape)
print('---------------')


target_action_df = pd.read_csv(TARGET_ACTION_FILE_PATH, sep='\t')
target_action_df['ymd'] = pd.to_datetime(target_action_df['ymd'].astype('str'))

print('target action')
print(target_action_df.columns)
print(target_action_df.head(5))

target_user_id_action_list = target_action_df['user_id'].unique()
target_session_id_action_list = target_action_df['session_id'].unique()

print('---------------')
print('number of users (target action data): {}'.format(len(target_user_id_action_list)))
print('number of sessions (target action data): {}'.format(len(target_session_id_action_list)))
print('dataframe size (train booking data)')
print(target_action_df.shape)
print('---------------')


# check how many users who do not have action information
# user_id_no_action_list = []
# print('---------------')
# print('user id (train)')
# for user_id in train_user_id_list:
#     if user_id not in train_user_id_action_list:
#         user_id_no_action_list.append(user_id)
# print('{} users (train) with no action information'.format(len(user_id_no_action_list)))
#
# check how many sessions which do not have action information
# session_id_no_action_list = []
# print('---------------')
# print('session id')
# for session_id in train_session_id_list:
#     if session_id not in train_session_id_action_list:
#         session_id_no_action_list.append(session_id)
# print('{} sessions (train) with no action information'.format(len(session_id_no_action_list)))


train_user_df =  pd.merge(train_booking_df, train_action_df, on=['ymd', 'user_id', 'session_id'])
print('train user df shape')
print(train_user_df.shape)
# train_user_df =  pd.merge(train_booking_df, train_action_df, on=['session_id'])
# print(train_user_df.shape)
print(train_user_df.columns)
print(train_user_df.head(5))

target_user_df =  pd.merge(target_booking_df, target_action_df, on=['ymd', 'user_id', 'session_id'])
print('target user df shape')
print(target_user_df.shape)
# train_user_df =  pd.merge(train_booking_df, train_action_df, on=['session_id'])
# print(train_user_df.shape)
print(target_user_df.columns)
print(target_user_df.head(5))


# print('---------------')
# print(train_user_df[train_user_df['user_id']==388309106223940])


# --------
# Naive approach
feature_columns = ['referer_code', 'is_app', 'agent_id', 'traffic_type', 'action_id', 'reference', 'step']
target_column = ['has_booking']

train_data_df = train_user_df[feature_columns + target_column]

print('train data')
print(train_data_df.shape)

train_data_df = train_data_df.reindex(np.random.permutation(train_data_df.index))

train_x = train_data_df[feature_columns]
train_y = train_data_df[target_column].values

train_sub_x, val_x, train_sub_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)


print('train')
print(train_x.shape)
print(train_y.shape)

test_x = target_user_df[feature_columns]
# test_y = target_user_df[target_column]

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


def train_xgb(X_train, Y_train, hyperparameter_tuning=False, nthread=4, folds=3, param_comb=5):
    """
    Train a xgb model
    Reference
    https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
    """

    xgb_clf = XGBClassifier(learning_rate=0.02,
                        n_estimators=100,
                        objective='binary:logistic',
                        silent=True, nthread=nthread)

    if hyperparameter_tuning:
        params = {
            'min_child_weight': [1, 5, 10],
            # 'gamma': [0.5, 1, 1.5, 2, 5],
            'gamma': [0.5, 1, 1.5],
            # 'subsample': [0.6, 0.8, 1.0],
            'subsample': [0.6, 0.8],
            # 'colsample_bytree': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8],
            # 'max_depth': [3, 4, 5]
            'max_depth': [4, 6],
        }

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        random_search = RandomizedSearchCV(xgb_clf, param_distributions=params, n_iter=param_comb, scoring='roc_auc',
                                           n_jobs=nthread,
                                           cv=skf.split(X_train, Y_train),
                                           verbose=3, random_state=1001)

        start_time = timer(None)
        random_search.fit(X_train, Y_train)
        timer(start_time)

        print('\n All results:')
        print(random_search.cv_results_)
        print('\n Best estimator:')
        print(random_search.best_estimator_)
        print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
        print(random_search.best_score_ * 2 - 1)
        print('\n Best hyperparameters:')
        print(random_search.best_params_)
        results = pd.DataFrame(random_search.cv_results_)
        results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

        xgb_clf = random_search
    else:
        xgb_clf.fit(train_sub_x, train_sub_y)

    xgb_model_path = 'xgb.model'
    xgb_clf.save_model(xgb_model_path)
    # joblib.dump(xgb_clf, xgb_model_path)
    print('save xgb model to {}'.format(xgb_model_path))

    return xgb_clf, xgb_model_path


def predict_xgb(xgb_model_path, X_test):
    dtest_mat = xgb.DMatrix(X_test)
    xgb_clf = xgb.Booster()
    xgb_clf.load_model(xgb_model_path)
    y_pred = xgb_clf.predict(dtest_mat)

    return y_pred


xgb_clf, xgb_model_path = train_xgb(train_sub_x, train_sub_y, hyperparameter_tuning=True)

y_pred = xgb_clf.predict(val_x)
# y_pred = predict_xgb(xgb_model_path, val_x)

print('y pred shape')
print(y_pred.shape)

print(y_pred)

# --------
# Feature engineering


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

y_true = val_y
print('y true shape')
print(y_true.shape)

print(y_true)

score = matthews_corrcoef(y_true, y_pred)
print(score)


