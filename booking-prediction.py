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
warnings.simplefilter("ignore", DeprecationWarning)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, classification_report, confusion_matrix
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

np.random.seed(42)

# ---
# Define file paths
TRAIN_BOOKING_FILE_PATH = 'data/case_study_bookings_train.csv'    # training sessions for bookings
TARGET_BOOKING_FILE_PATH = 'data/case_study_bookings_target.csv'  # target sessions to predict bookings

TRAIN_ACTION_FILE_PATH = 'data/case_study_actions_train.csv'       # training set of user actions
TARGET_ACTION_FILE_PATH = 'data/case_study_actions_target.csv'     # user actions in the target sessions

# replace the NAN values by a specific value
NA_ACTION_ID = -10
NA_REFERENCE_ID = -10
NA_STEP = 0

feature_columns = ['referer_code', 'is_app', 'agent_id', 'traffic_type', 'action_id', 'reference', 'step']
target_column = ['has_booking']


def preprocessing(df):
    df['action_id'].fillna(NA_ACTION_ID, inplace=True)
    df['reference'].fillna(NA_REFERENCE_ID, inplace=True)
    df['step'].fillna(NA_STEP, inplace=True)

    df['referer_code'] = df['referer_code'].astype('int')
    df['is_app'] = df['is_app'].astype('int')
    df['agent_id'] = df['agent_id'].astype('int')
    df['traffic_type'] = df['traffic_type'].astype('int')
    df['action_id'] = df['action_id'].astype('int')
    df['reference'] = df['reference'].astype('int')
    df['step'] = df['step'].astype('int')

    if 'has_booking' in df.columns:
        df['has_booking'] = df['has_booking'].astype('int')

    return df

def get_train_set(df):
    train_df = df[feature_columns + target_column]

    train_x = train_df[feature_columns]
    train_y = train_df[target_column].values

    # https://stackoverflow.com/questions/31995175/scikit-learn-cross-val-score-too-many-indices-for-array
    """
    When we do cross validation in scikit-learn, the process requires an (R,) shape label instead of (R,1). 
    Although they are the same thing to some extend, their indexing mechanisms are different. So in your case, just add:
    c, r = labels.shape
    labels = labels.reshape(c,)
    """
    c, r = train_y.shape
    train_y = train_y.reshape(c, )

    return train_x, train_y

def get_test_set(df):
    test_x = df[feature_columns]

    return test_x

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

    xgb_clf = XGBClassifier(n_estimators=100, nthread=n_jobs, objective='binary:logistic', silent=True,)

    if hyperparameter_tuning:
        print('xgb hyperparameter tuning ...')

        params = {
            'n_estimators': [80, 100, 200, 300],
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

        xgb_clf = random_search
        #xgb_clf = random_search.best_estimator_
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
    print('save the xgb model to {}'.format(xgb_model_path))

    return xgb_clf, xgb_model_path


def train_rf(X_train, Y_train, hyperparameter_tuning=False, model_path=None, n_jobs=3, folds=3):
    """
    Train a RF classifier

    Reference
    https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=n_jobs)

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
    print('save the rf model to {}'.format(model_path))

    return model, model_path


def train_nb(X_train, Y_train, model_path=None):
    """
    Train a naive bayes classifier
    """
    # reference https://www.analyticsvidhya.com/blog/2017/09/naive-bayes-explained/
    model = GaussianNB()
    model.fit(X_train, Y_train,)

    if model_path is None:
        model_path = 'nb.model'

    joblib.dump(model, model_path)
    print('save the GaussianNB model to {}'.format(model_path))

    return model, model_path


def train_lgbm(X_train, Y_train, categorical_feature=[0, 1, 2, 3, 4, 5],
               model_path=None, n_jobs=3, hyperparameter_tuning=False, num_boost_round=100, folds=3):
    """
    Train a lightGBM model

    Reference
    https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py
    https://www.kaggle.com/garethjns/microsoft-lightgbm-with-parameter-tuning-0-823?scriptVersionId=1751960
    """
    d_train = lgb.Dataset(X_train, label=Y_train,
                          # categorical_feature=['aisle_id', 'department_id']
                          categorical_feature=categorical_feature,
                          )


    if not hyperparameter_tuning:
        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'num_class': 1,                # must be 1 for non-multiclass training
            'metric': 'binary_error',
            #'metric': 'binary_logloss',
            #'n_jobs': n_jobs,
            'nthread': n_jobs,
            #'num_leaves': 31,
            'num_leaves': 64,
            'min_child_weight': 1,
            'min_child_samples': 5,
            'scale_pos_weight': 1,
            'reg_alpha': 5,
            'learning_rate': 0.05,
            'max_bin': 512,
            #'feature_fraction': 0.9,
            #'bagging_fraction': 0.8,
            #'bagging_freq': 5,
            #'verbose': 0
        }

        gbm = lgb.train(params,
                        d_train,
                        num_boost_round=num_boost_round,
                        categorical_feature=categorical_feature)

    else:
        params = {'boosting_type': 'gbdt',
                  'max_depth': -1,
                  'objective': 'binary',
                  'nthread': n_jobs,  # Updated from nthread
                  'num_leaves': 64,
                  'learning_rate': 0.05,
                  'max_bin': 512,
                  'subsample_for_bin': 200,
                  'subsample': 1,
                  'subsample_freq': 1,
                  'colsample_bytree': 0.8,
                  'reg_alpha': 5,
                  'reg_lambda': 10,
                  'min_split_gain': 0.5,
                  'min_child_weight': 1,
                  'min_child_samples': 5,
                  'scale_pos_weight': 1,
                  'num_class': 1,
                  'metric': 'binary_error'}

        gridParams = {
            'learning_rate': [0.005],
            'n_estimators': [8, 16, 24],
            'num_leaves': [6, 8, 12, 16],
            'boosting_type': ['gbdt'],
            'objective': ['binary'],
            'random_state': [42],  # Updated from 'seed'
            'colsample_bytree': [0.64, 0.65, 0.66],
            'subsample': [0.7, 0.75],
            'reg_alpha': [1, 1.2],
            'reg_lambda': [1, 1.2, 1.4],
        }

        mdl = lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 n_jobs=n_jobs,  # Updated from 'nthread'
                                 silent=True,
                                 max_depth=params['max_depth'],
                                 max_bin=params['max_bin'],
                                 subsample_for_bin=params['subsample_for_bin'],
                                 subsample=params['subsample'],
                                 subsample_freq=params['subsample_freq'],
                                 min_split_gain=params['min_split_gain'],
                                 min_child_weight=params['min_child_weight'],
                                 min_child_samples=params['min_child_samples'],
                                 scale_pos_weight=params['scale_pos_weight'])

        print(mdl.get_params().keys())

        grid = GridSearchCV(mdl, gridParams, verbose=1, cv=folds, n_jobs=n_jobs)
        grid.fit(X_train, Y_train)

        print('best parameters:')
        print(grid.best_params_)
        print('best score: ')
        print(grid.best_score_)

        # using parameters already set above, replace in the best from the grid search
        params['colsample_bytree'] = grid.best_params_['colsample_bytree']
        params['learning_rate'] = grid.best_params_['learning_rate']
        params['max_bin'] = grid.best_params_['max_bin']
        params['num_leaves'] = grid.best_params_['num_leaves']
        params['reg_alpha'] = grid.best_params_['reg_alpha']
        params['reg_lambda'] = grid.best_params_['reg_lambda']
        params['subsample'] = grid.best_params_['subsample']
        params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

        print('Fitting with params: ')
        print(params)

        gbm = lgb.train(params,
                        X_train,
                        1000,
                        #valid_sets=[trainDataL, validDataL],
                        #early_stopping_rounds=50,
                        verbose_eval=4)

        # Plot importance
        #lgb.plot_importance(gbm)

    if model_path is None:
        model_path = 'lgbm.model'
        if hyperparameter_tuning:
            model_path = 'lgbm.ht.model'

    # save model to file
    gbm.save_model(model_path)
    print('save the lightGBM model to {}'.format(model_path))

    # load model to predict
    # print('Load model to predict')
    # bst = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    # y_pred = bst.predict(X_test)

    return gbm, model_path


def train_catboost(X_train, Y_train, categorical_feature=[0, 1, 2, 3, 4, 5],
               model_path=None, hyperparameter_tuning=False, num_boost_round=100):
    """
    train a catboost model
    """
    model = CatBoostClassifier(loss_function='Logloss',
                               iterations=num_boost_round,
                               #learning_rate=1,
                               #depth=2
                               )
    model.fit(X_train, Y_train, categorical_feature)

    if model_path is None:
        model_path = 'catboost.model'
        if hyperparameter_tuning:
            model_path = 'catboost.ht.model'

    model.save_model(model_path)

    print('save the catboost model to {}'.format(model_path))

    return model, model_path



def predict(model_path, X_test, is_lgbm=False, is_catboost=False):
    """
    load the model and predict unseen data
    """
    if is_lgbm:
        # lightgbm
        model = lgb.Booster(model_file=model_path)
    elif is_catboost:
        model = CatBoostClassifier()
        model = model.load_model(model_path)
    else:
        # sklearn
        # xgboost
        model = joblib.load(model_path)

    # y_pred = model.predict_prob(X_test)
    y_pred = model.predict(X_test)

    if is_lgbm:
        return np.array([np.argmax(y) for y in y_pred])
    else:
        return y_pred


def blend_predictions(y_pred_list, threshold=0.7):
    """
    blend the predictions
    """
    y_pred = y_pred_list[0]

    for i in range(1, len(y_pred_list)):
        for j in range(len(y_pred)):
            y_pred[j] += y_pred_list[i][j]

    y_pred = y_pred*1.0 / len(y_pred_list)

    y_output = []
    for y in y_pred:
        if y > threshold:
            y_output.append(1)
        else:
            y_output.append(0)

    return np.array(y_output)


if __name__ == "__main__":

    # -------------------
    # Step 1: read and explore the data
    #
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

    train_user_df = pd.merge(train_booking_df, train_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

    train_user_df = preprocessing(train_user_df)

    target_user_df = pd.merge(target_booking_df, target_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

    target_user_df = preprocessing(target_user_df)

    # shuffle
    #train_data_df = train_data_df.reindex(np.random.permutation(train_data_df.index))

    train_x, train_y = get_train_set(train_user_df)
    print('-------------------')
    print('train set size:')
    print(train_x.shape)
    print(train_y.shape)

    train_sub_x, val_x, train_sub_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    test_x = get_test_set(target_user_df)
    print('-------------------')
    print('test set size:')
    print(test_x.shape)
    # print(test_y.shape)


    # -------------------
    # Step 2: feature engineering
    #
    # category data in machine learning
    # Reference:
    # https://medium.com/unstructured/how-feature-engineering-can-help-you-do-well-in-a-kaggle-competition-part-i-9cc9a883514d
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-66041f734512
    # https://blog.myyellowroad.com/using-categorical-data-in-machine-learning-with-python-from-dummy-variables-to-deep-category-42fd0a43b009


    # -------------------
    # Step 3: train model and make predictions
    #
    y_pred_list = []

    # model, model_path = train_xgb(train_sub_x, train_sub_y, hyperparameter_tuning=True, model_path='xgb.ht.model')
    # y_pred = predict('xgb.ht.model', val_x)
    # y_pred_list.append(y_pred)

    # model, model_path = train_xgb(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='xgb.model')
    # y_pred = predict('xgb.model', val_x)
    # y_pred_list.append(y_pred)

    model, model_path = train_lgbm(train_sub_x, train_sub_y, hyperparameter_tuning=True, model_path='lgbm.ht.model', num_boost_round=100)
    y_pred = predict('lgbm.ht.model', val_x, is_lgbm=True)
    y_pred_list.append(y_pred)

    # model, model_path = train_catboost(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='catboost.model', num_boost_round=200)
    # y_pred = predict(model_path='catboost.model', X_test = val_x, is_catboost=True)
    # y_pred_list.append(y_pred)
    #
    # model, model_path = train_rf(train_sub_x, train_sub_y, hyperparameter_tuning=False, model_path='rf.model')
    # y_pred = predict('rf.model', val_x)
    # y_pred_list.append(y_pred)
    #
    # model, model_path = train_nb(train_sub_x, train_sub_y, model_path='nb.model')
    # y_pred = predict('nb.model', val_x)
    # y_pred_list.append(y_pred)

    y_pred = blend_predictions(y_pred_list)
    print(y_pred)
    print(len(y_pred))
    print(type(y_pred))


    # -------------------
    # Step 4: evaluate the model
    #
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

    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn)
    print(fp)
    print(fn)
    print(tp)
    print('---')
    
    mcc = (tp*tn - fp*fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print(mcc)
    print(matthews_corrcoef(y_true, y_pred))
    """

    mcc_score = matthews_corrcoef(y_true, y_pred)
    print('matthews corrcoef score {}'.format(mcc_score))

    accuracy = accuracy_score(y_true, y_pred)
    print('accuracy: {}'.format(accuracy))
    print('classification report:')
    print(classification_report(y_true, y_pred))


