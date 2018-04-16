import time
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

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
    default_action_values = [-10, -10]
    previous_action_names = ['action_id', 'reference']

    def __init__(self, df, nb_previous_action, n_jobs=4):
        self.df = df
        self.nb_previous_action = nb_previous_action
        self.n_jobs = n_jobs

    def func_add_previous_action(self, session_id):
        current_steps = self.df[self.df['session_id'] == session_id]['step'].tolist()

        for current_step in current_steps:
            for i in range(self.nb_previous_action):
                previous_step = current_step - (i + 1)
                previous_df = self.df[(self.df['session_id'] == session_id) & (self.df['step'] == previous_step)]
                if not previous_df is None:
                    for previous_action in self.previous_action_names:
                        col_name = '{}_{}'.format(previous_action, (i + 1))
                        self.df.ix[(self.df['session_id'] == session_id) & (self.df['step'] == current_step), col_name] = previous_df[previous_action]

        print('---')
        print(session_id)
        print(self.df[self.df['session_id']==session_id])


    def add_previous_action(self,):
        """
        add previous action information (e.g., action_id, reference) to the dataframe
        """
        # TODO: this function should be optimized, e.g., using mapping function.

        print('== add previous action information ==')

        for i in range(self.nb_previous_action):
            for j, previous_action in enumerate(self.previous_action_names):
                new_col_name = '{}_{}'.format(previous_action, (i + 1))
                self.df[new_col_name] = self.default_action_values[j]

        start_time = time.time()

        session_id_list = self.df['session_id'].unique()
        for k, session_id in enumerate(session_id_list):
            current_steps = self.df[self.df['session_id'] == session_id]['step'].tolist()

            for current_step in current_steps:
                for i in range(self.nb_previous_action):
                    previous_step = current_step - (i + 1)
                    previous_df = self.df[(self.df['session_id'] == session_id) & (self.df['step'] == previous_step)]
                    if (not previous_df is None) and (not previous_df.empty):
                        for previous_action in self.previous_action_names:
                            col_name = '{}_{}'.format(previous_action, (i + 1))
                            # print('previous')
                            # print(previous_df[previous_action].values)
                            # print('----')
                            self.df.ix[(self.df['session_id'] == session_id) & (self.df['step'] == current_step), col_name] = previous_df[previous_action].values

            if k % 2 == 0:
                time_used = time.time() - start_time
                time_needed = (time_used / (k + 1)) * (len(session_id_list) - k - 1)

                print('{} / {}'.format(k + 1, len(session_id_list)))
                print('time used (mins): {}'.format(round(time_used / 60, 2)))
                print('time required (mins): {}'.format(round(time_needed / 60, 2)))

                print('\nnew dataframe')
                print(self.df.head(5))

                print('---')
                print(session_id)
                print(self.df[self.df['session_id'] == session_id])

            # Parallel(n_jobs=self.n_jobs)(delayed(self.func_add_previous_action)(session_id) for session_id in session_id_list)

            # print(self.df.head(5))
            # start_time = time.time()
            # if k % 2 == 0:
            #     time_used = time.time() - start_time
            #     time_needed = (time_used / (k + 1)) * (len(session_id_list) - k - 1)
            #
            #     print('{} / {}'.format(k + 1, len(session_id_list)))
            #     print('time used (mins): {}'.format(round(time_used / 60, 2)))
            #     print('time required (mins): {}'.format(round(time_needed / 60, 2)))
            #
            #     print('\nnew dataframe')
            #     print(self.df.head(5))

            # current_steps = df[df[session_id_name] == session_id][step_name].tolist()
            # for current_step in current_steps:
            #     for i in range(nb_previous_action):
            #         previous_step = current_step - (i + 1)
            #         previous_df = df[(df[session_id_name] == session_id) & (df[step_name] == previous_step)]
            #         if not previous_df is None:
            #             for previous_action in previous_action_list:
            #                 col_name = '{}_{}'.format(previous_action, (i + 1))
            #                 df[(df[session_id_name] == session_id) & (df[step_name] == current_step)][col_name] = \
            #                 previous_df[previous_action]



if __name__ == "__main__":

    # ----------
    # read data
    train_booking_df = pd.read_csv(TRAIN_BOOKING_FILE_PATH, sep='\t')
    train_booking_df['ymd'] = pd.to_datetime(train_booking_df['ymd'].astype('str'))

    target_booking_df = pd.read_csv(TARGET_BOOKING_FILE_PATH, sep='\t')
    target_booking_df['ymd'] = pd.to_datetime(target_booking_df['ymd'].astype('str'))

    train_action_df = pd.read_csv(TRAIN_ACTION_FILE_PATH, sep='\t')
    train_action_df['ymd'] = pd.to_datetime(train_action_df['ymd'].astype('str'))

    target_action_df = pd.read_csv(TARGET_ACTION_FILE_PATH, sep='\t')
    target_action_df['ymd'] = pd.to_datetime(target_action_df['ymd'].astype('str'))

    train_user_df = pd.merge(train_booking_df, train_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

    train_user_df = preprocessing(train_user_df)

    target_user_df = pd.merge(target_booking_df, target_action_df, on=['ymd', 'user_id', 'session_id'], how='left')

    target_user_df = preprocessing(target_user_df)

    # ----------
    # add previous action information
    addprevAc = AddPreActions(df=train_user_df, nb_previous_action=2, n_jobs=6)
    addprevAc.add_previous_action()
    train_user_df = addprevAc.df

    print(train_user_df.head(10))

    df_path = 'train_user_df_{}'.format(2)
    train_user_df.to_csv(df_path, sep='\t')