# Prediction of bookings based on user behavior

Data Scientist – User Profiling, Hotel Search

## Situation:

A search session describes a user’s journey to find his ideal hotel, by including all his interactions. Given user search sessions, we are interested in predicting the outcome of these sessions based on the users’ interactions; as well determining which of these interactions have the highest importance for this estimation.


## Data:

We provide two kinds of data sets:

* anonymized user logs generated by usage on our website (user actions); and,
* booking outcome per session with contextual information (bookings).

Both types of datasets are split by the same timestamp into train and target sets. The target set contains the same information as the training set, except the outcome (i.e. has_booking). More information is provided in README.md (You can find this in the resources section).


## Task:

The task is to train a machine learning model to estimate if a booking occurred – the training and target sets have been provided for you. We expect binary predictions for the target sessions, which will be evaluated by Matthews Correlation Coefficient (MCC) using the ground truth dataset on our side. You can have as many submissions as you would like to improve your solution.


## Additional questions:

* What makes the classification problem difficult in this task? How do you handle that?
* Evaluate and compare at least 3 classification methods for this task.
* Propose at least 3 features that are significant to predict bookings?
* We can spot a very significant action type. What might this action refer to?


## Files

* notebook:
    - booking_prediction.ipynb
    - booking_prediction-v2.ipynb

* scripts:
    - booking-prediction.py
    - booking-prediction-add-previous-action-features.py
    - booking-prediction-data-preparation.py
    - booking-prediction-v2.py

* predictions:
    - prediction-*.csv

