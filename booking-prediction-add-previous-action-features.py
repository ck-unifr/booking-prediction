# Prediction of bookings based on user behavior
# Data Scientist – User Profiling, Hotel Search
#
# Feature engineering
#
# Author: Kai Chen
# Date: Apr, 2018
#

import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import _thread
from multiprocessing import Pool
from random import shuffle
import gc

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
    print('\n === preprocess data === \n')

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


def get_train_set(df, feature_columns):
    print('\n === get train set === \n')

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


def get_test_set(df, feature_columns):
    print('\n === get test set === \n')

    test_x = df[feature_columns]

    return test_x



class AddPreActions:
    """
    for each session at each step, add its previous steps information
    """
    previous_action_names = ['action_id', 'reference']
    default_action_values = [-10, -10]

    def __init__(self, df, nb_previous_action=2, step_size=100, n_jobs=2):
        self.df = df
        self.nb_previous_action = nb_previous_action
        self.n_jobs = n_jobs
        self.step_size = step_size

    def func_add_previous_action(self, session_id):
        """
        for each session, at each step_size add nb_previous_action previous action information
        """
        current_steps = self.df[self.df['session_id'] == session_id]['step'].tolist()

        # TODO: use groupby
        # for current_step in current_steps:
        for j in range(0, len(current_steps), self.step_size):
            current_step = current_steps[j]

            # add nb_previous_action previous action information
            for i in range(self.nb_previous_action):
                previous_step = current_step - (i + 1)
                previous_df = self.df[(self.df['session_id'] == session_id) & (self.df['step'] == previous_step)]

                if (not previous_df is None) and (not previous_df.empty):
                    for previous_action in self.previous_action_names:
                        col_name = '{}_{}'.format(previous_action, (i + 1))
                        # print('previous')
                        # print(previous_df[previous_action].values)
                        # print('----')
                        self.df.loc[(self.df['session_id'] == session_id) & (self.df['step'] == current_step), col_name] = previous_df[previous_action].values[0]

                    del previous_df
                    gc.collect()

        del current_steps
        gc.collect()
        # print('---')
        # print(session_id)
        # print(self.df[self.df['session_id'] == session_id])

    def add_previous_action(self,  steps = 5000):
        """
        for each session, at each step_size steps, add nb_previous_action previous action information
        """
        print('== add previous action information ==')

        for i in range(self.nb_previous_action):
            for j, previous_action in enumerate(self.previous_action_names):
                new_col_name = '{}_{}'.format(previous_action, (i + 1))
                self.df[new_col_name] = self.default_action_values[j]

        # start_time = time.time()
        session_id_list = self.df['session_id'].unique()

        print('\nnumber of sessions: {}'.format(len(session_id_list)))

        start = 0
        start_time = time.time()
        session_len = len(session_id_list)
        while start < session_len:
            end = start + steps
            if end > session_len:
                end = session_len
            print('start: {} end: {}'.format(start, end))
            with Pool(self.n_jobs) as p:
                p.map(self.func_add_previous_action, session_id_list[start:end])
            start = end

            time_used = time.time() - start_time
            time_needed = (time_used / (end))*(session_len - end)

            print('{}/{}'.format(end, session_len))
            print('time used (mins): {}'.format(round(time_used/60, 2)))
            print('time required (mins): {}'.format(round(time_needed / 60, 2)))


def get_data(df, percentage, csv_path):
    """
    randomly select a part of the data and save it to a csv file
    """
    if percentage >= 100:
        return df
    else:
        print('full data shape: ')
        print(df.shape)

        df = df.reindex(np.random.permutation(df.index))
        df_sub = df[0:int((percentage/100.0)*(df.shape[0]))]

        df_sub.to_csv(csv_path)
        print('sub data shape: ')
        print(df_sub.shape)



if __name__ == "__main__":

    # ----------
    # read train data
    # train_booking_df = pd.read_csv(TRAIN_BOOKING_FILE_PATH, sep='\t')
    # train_booking_df['ymd'] = pd.to_datetime(train_booking_df['ymd'].astype('str'))
    # train_action_df = pd.read_csv(TRAIN_ACTION_FILE_PATH, sep='\t')
    # train_action_df['ymd'] = pd.to_datetime(train_action_df['ymd'].astype('str'))
    # train_user_df = pd.merge(train_booking_df, train_action_df, on=['ymd', 'user_id', 'session_id'], how='left')
    # train_user_df = preprocessing(train_user_df)
    # train_user_df.to_csv('train_user_df.csv')


    # due to memory issue, take a subset of the data
    # percentage = 100
    # train_user_df = get_data(train_user_df, percentage, 'train_user_df_{}.csv'.format(percentage))

    # ----------
    # read test data
    # target_booking_df = pd.read_csv(TARGET_BOOKING_FILE_PATH, sep='\t')
    # target_booking_df['ymd'] = pd.to_datetime(target_booking_df['ymd'].astype('str'))
    # target_action_df = pd.read_csv(TARGET_ACTION_FILE_PATH, sep='\t')
    # target_action_df['ymd'] = pd.to_datetime(target_action_df['ymd'].astype('str'))
    # target_user_df = pd.merge(target_booking_df, target_action_df, on=['ymd', 'user_id', 'session_id'], how='left')
    # target_user_df = preprocessing(target_user_df)
    # target_user_df.to_csv('target_user_df.csv')

    # ----------
    # add previous action information
    # for each session, at each step_size add nb_previous_action previous action information
    step_size = 200
    print('step size: {}'.format(step_size))
    for nb_previous_action in [3, 4]:
        train_user_df = pd.read_csv('train_user_df.csv')
        target_user_df = pd.read_csv('target_user_df.csv')

        # shuffle the train set
        train_user_df = train_user_df.reindex(np.random.permutation(train_user_df.index))

        print('----------')
        print('\nnumber of previous steps: {}'.format(nb_previous_action))
        # train data
        print('\n{}'.format(nb_previous_action))

        print('\ntrain data')
        addprevAc = AddPreActions(df=train_user_df, nb_previous_action=nb_previous_action, step_size=step_size, n_jobs=2)
        addprevAc.add_previous_action()
        train_user_df = addprevAc.df

        df_path = 'train_user_df_{}_{}.csv'.format(step_size, nb_previous_action)
        # if percentage < 100:
        #     df_path = 'train_user_df_{}_{}_{}.csv'.format(step_size, nb_previous_action, percentage)
        train_user_df.to_csv(df_path, sep='\t')
        print('\nsave train data to {}'.format(df_path))
        # print(train_user_df.head(5))

        print('\ntarget data')
        addprevAc = AddPreActions(df=target_user_df, nb_previous_action=nb_previous_action, step_size=step_size, n_jobs=2)
        addprevAc.add_previous_action()
        target_user_df = addprevAc.df

        df_path = 'target_user_df_{}_{}.csv'.format(step_size, nb_previous_action)
        target_user_df.to_csv(df_path, sep='\t')
        print('\nsave target data to {}'.format(df_path))
        # print(target_user_df.head(5))

        del train_user_df
        del target_user_df
        gc.collect()