









import random
import datetime
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tqdm.auto import tqdm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# load the data from the source file
# only need the account_id (a primary key, customer identifier) and the date columns
# convert the date column from strings like "930101" into a proper datetime format
df = pd.read_csv(filepath_or_buffer='trans.zip',
                 usecols=['account_id', 'date'],
                 parse_dates=['date'])

# define the training (calibration) and prediction (holdout) period

training_start = '1993-01-01'
training_end   = '1995-12-31'
holdout_start  = '1996-01-01'
holdout_end    = '1998-12-31'
date_format    = '%Y-%m-%d'

# display basic stats

cohort_accounts = df.groupby('account_id').min().query(
    'date <= @training_end').reset_index()['account_id'].tolist()

df = df.query('account_id in @cohort_accounts')
df = df.sort_values(by='account_id').reset_index(drop=True)

print(f"Accounts in dataset:  {len(df['account_id'].unique())}")
print(f"Total transactions: {len(df)}")



aggregate_counts = df.copy(deep=True)
aggregate_counts['year'] = aggregate_counts['date'].dt.year
aggregate_counts['week'] = (aggregate_counts['date'].dt.dayofyear // 7).clip(upper=51) # we roll the 52nd week into the 51st
aggregate_counts = aggregate_counts.groupby(['year', 'week']).agg({'account_id': 'count', 'date': 'min'}).reset_index()