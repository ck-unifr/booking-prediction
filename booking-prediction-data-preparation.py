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
import random
import warnings

import gc

warnings.filterwarnings('ignore')
warnings.simplefilter("ignore", DeprecationWarning)

import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import product

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

feature_columns = ['ymd', 'referer_code', 'is_app', 'agent_id', 'traffic_type', 'action_id', 'reference', 'step']
# target_column = ['has_booking']

def prepare_data(df, nb_pre_steps=1,
                 feature_columns = ['ymd', 'referer_code', 'is_app', 'agent_id', 'traffic_type', 'action_id', 'reference', 'step'],
                 previous_action_names=['action_id', 'reference'],
                 target_column = 'has_booking',
                 default_action_values = [-10, -10]):
    """
    create a dataframe, such that each row contains the information of a session.
    Since for each session, there is a sequence of information.
    In this dataframe, for each session,
    I take only the last step information with its nb_pre_steps number of previous steps information.
    """
    print('\n== prepare data ==')

    total_nb_rows = len(df['session_id'].unique())

    # initialize the column names
    columns_add = ['duration'] # add new features
    columns_add = columns_add + [f_name for f_name in feature_columns]

    for i in range(0, nb_pre_steps):
        for previous_action_name in previous_action_names:
            col_name = '{}_{}'.format(previous_action_name, (i+1))
            columns_add.append(col_name)

    if target_column in df.columns:
        columns_add.append(target_column)

    df_new = pd.DataFrame(columns=columns_add)

    start_time = time.time()
    index = 0 # index of each row
    for name, group in df.groupby('session_id'):
        max_step = np.max(group['step'])
        min_step = np.min(group['step'])

        # get start time
        start_time = pd.to_datetime(group[group['step'] == max_step]['ymd'].values[0].astype('str'))

        # get end time
        end_time = pd.to_datetime(group[group['step'] == min_step]['ymd'].values[0].astype('str'))

        # compute the duration of the session
        duration = (end_time-start_time).total_seconds()

        # for each session, get its information in the last step
        sub_df = group[group['step'] == max_step]

        # set the initial values of this session
        val_add = []

        # duration
        val_add.append(duration)

        for feature_column in feature_columns:
            val_add.append(sub_df[feature_column].values[0])

        for i in range(0, nb_pre_steps):
            for j, previous_action_name in enumerate(previous_action_names):
                val_add.append(default_action_values[j])

        if target_column in sub_df.columns:
            val_add.append(sub_df[target_column].values[0])

        df_new = df_new.append(pd.DataFrame([val_add], columns=columns_add))

        # get the session previous steps information and add it to the new row
        for i in range(0, nb_pre_steps):
            step = max_step - i - 1
            sub_df = group[group['step'] == step]
            if (not sub_df is None) and (not sub_df.empty):
                for previous_action_name in previous_action_names:
                    col_name = '{}_{}'.format(previous_action_name, step)
                    # print('previous')
                    # print(previous_df[previous_action].values)
                    # print('----')
                    df_new.iloc[index][col_name] = sub_df[previous_action_name].values[0]


        index += 1

        if index % 20000 == 0:
            time_used = time.time() - start_time
            time_needed = time_used / index * (total_nb_rows-index)
            print('\n{} / {}'.format(index, total_nb_rows))
            print('time used (mins): {}'.format(round(time_used / 60)))
            print('time needed (mins): {}'.format(round(time_needed / 60)))
            print(df_new.iloc[random.randint(0, index-1)][columns_add])
            if target_column in df_new.columns:
                print('{}: {}'.format(target_column, df_new.iloc[random.randint(0, index-1)][target_column]))

    return df_new

def prepare_datasets(param_dict):
    train_user_df = param_dict['train']
    target_user_df = param_dict['target']
    nb_prev_step = param_dict['nb_prev_step']

    print('\n{}'.format(nb_prev_step))

    train_user_df_new = prepare_data(train_user_df, nb_pre_steps=nb_prev_step)
    df_path = '{}-{}.csv'.format('train_user_df', nb_prev_step)
    train_user_df_new.to_csv(df_path, index=False)
    print('\nsave train dataframe to {}'.format(df_path))
    print(train_user_df_new.head(2))

    target_user_df_new = prepare_data(target_user_df, nb_pre_steps=nb_prev_step)
    df_path = '{}-{}.csv'.format('target_user_df', nb_prev_step)
    target_user_df_new.to_csv(df_path, index=False)
    print('\nsave test dataframe to {}'.format(df_path))
    print(target_user_df_new.head(2))

    del train_user_df_new
    del target_user_df_new
    gc.collect()


if __name__ == "__main__":

    # train_user_df = pd.read_csv('train_user_df.csv')
    # target_user_df = pd.read_csv('target_user_df.csv')
    #
    # train_user_df = train_user_df.drop(columns=['Unnamed: 0'])
    # train_user_df.to_csv('train_user_df.csv', index=False)
    #
    # target_user_df = target_user_df.drop(columns=['Unnamed: 0'])
    # target_user_df.to_csv('target_user_df.csv', index=False)

    train_user_df = pd.read_csv('train_user_df.csv')
    train_user_df['ymd'] = pd.to_datetime(train_user_df['ymd'].astype('str'))
    target_user_df = pd.read_csv('target_user_df.csv')
    target_user_df['ymd'] = pd.to_datetime(target_user_df['ymd'].astype('str'))

    print(train_user_df.shape)

    print(train_user_df.head(3))

    print(target_user_df.head(3))


    # nb_prev_step_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    nb_prev_step_list = [1, 2, 4]
    param_dict_list = []
    for nb_prev_step in nb_prev_step_list:
        param_dict = dict()
        param_dict['train'] = train_user_df
        param_dict['target'] = target_user_df
        param_dict['nb_prev_step'] = nb_prev_step
        param_dict_list.append(param_dict)

    n_jobs = 2
    with Pool(n_jobs) as p:
        p.map(prepare_datasets, param_dict_list)






