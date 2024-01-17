import numpy as np
import pandas as pd
import collections
import math

from surprise import Reader
from surprise.accuracy import rmse
from surprise import Dataset
from surprise.prediction_algorithms.matrix_factorization import SVD

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
from stable_baselines.deepq.policies import FeedForwardPolicy
import tensorflow as tf

from environment_al import rec_env_AL
from environment_items import rec_env_items

# Import data

df = pd.read_csv("useritemmatrix.csv")
df.drop(["Unnamed: 0"], axis=1, inplace=True)

# Transform into categorical codes
df["itemId"] = df["itemId"].astype("category").cat.codes

# Take a sample
df = df.sample(n=200000, random_state=42)

# checking users who have many purchases
user_freq_df = pd.DataFrame.from_dict(collections.Counter(df['userId']), orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index': 'userId', 0: 'freq'})

# percentage of total number of users to set as cold user
perc_cold_users = 0.25
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))

# select the [nr_of_cold_users] with the highest number of interactions
cold_users = user_freq_df.sample(nr_of_cold_users, random_state=1)
cold_users = cold_users.get_value(index=range(0, (nr_of_cold_users)), col=0, takeable=True)

print('Selecting ' + str(nr_of_cold_users) + ' cold user(s)')

# compute purchase purchase/return frequency per item
item_freq_counter = collections.Counter(df['itemId'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter, orient='index'). reset_index()
item_freq_df = item_freq_df.rename(columns={'index': 'itemId', 0: 'freq'})

# produce list of items which are at least interacted with [threshold_item] times
threshold_item = 10
threshold_item_df = item_freq_df[item_freq_df['freq'] >= threshold_item]['itemId']

# Tuning the model

# tune = df[df['itemId'].isin(np.asarray(threshold_item_df))]
# tune = tune.sample(n=200000)

# from surprise.model_selection import GridSearchCV
# reader = Reader(rating_scale=(0, 1))
# data = Dataset.load_from_df(tune, reader)

# n_factors = [10, 50, 100, 150, 200, 300]
# lr_all = [0.001, 0.003, 0.005]
# reg_all = [0.01, 0.02, 0.05]
# n_epochs  = [50, 100, 150]


# param_grid = {'n_factors': n_factors,
#               'lr_all': lr_all,
#               'reg_all': reg_all,
#               'n_epochs': n_epochs }

# grid = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=10)

# grid.fit(data)

# # best RMSE score
# print(grid.best_score['rmse'])

# # combination of parameters that gave the best RMSE score
# print(grid.best_params['rmse'])

# Compute the Strategy Scores

# Gini Score

# function to compute Gini
def gini(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    gini = 0.

    sum_probs = 0

    for iterator in probs:
        sum_probs += iterator * iterator

    gini = 1 - sum_probs
    return gini


unique_itemId = pd.Series(threshold_item_df)
gini_list = np.zeros(shape=(len(unique_itemId), 2))
j = 0

# loop over all itemId's and compute the Gini for each item
for i in unique_itemId:
    item_i_df = df[df['itemId'] == i]
    gini_list[j] = [i, gini(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': gini_list[:, 0], 'gini': gini_list[:, 1]}
gini_items_df = pd.DataFrame(to_df)
gini_items_df.sort_values(by='gini', inplace=True, ascending=False)


del gini_list

print('Computed Gini scores for all items')

# PopGini Score

# prepare item gini scores for merging
gini_items_df2 = gini_items_df.sort_values(by='itemId')
gini_items_df2.set_index(keys='itemId', inplace=True)

# merge frequencies and entropies
popgini_items_df = pd.concat([item_freq_df, gini_items_df2], axis=1, join='inner')

# set weights for the popgini score
weight_popularity = 0.9
weight_gini = 1

# compute popgini score
popgini_items_df['popgini'] = weight_popularity*np.log10(popgini_items_df['freq'])+weight_gini*popgini_items_df['gini']
popgini_items_df.sort_values(by='popgini', inplace=True, ascending=False)

print('Computed PopGini scores for all items')


# Entropy Score

# function to compute entropy
def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    for iterator in probs:
        ent -= iterator * np.log2(iterator)

    return ent


unique_itemId = pd.Series(threshold_item_df)
entropy_list = np.zeros(shape=(len(unique_itemId), 2))
j = 0

# loop over all itemId's and compute the entropy for each item
for i in unique_itemId:
    item_i_df = df[df['itemId'] == i]
    entropy_list[j] = [i, entropy(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': entropy_list[:, 0], 'entropy': entropy_list[:, 1]}
ent_items_df = pd.DataFrame(to_df)
ent_items_df.sort_values(by='entropy', inplace=True, ascending=False)

del entropy_list

print('Computed entropy scores for all items')

# PopEnt Score

# prepare item entropies for merging
ent_items_df2 = ent_items_df.sort_values(by='itemId')
ent_items_df2.set_index(keys='itemId', inplace=True)

# merge frequencies and entropies
popent_items_df = pd.concat([item_freq_df, ent_items_df2], axis=1, join='inner')

# set weights for the popent score
weight_popularity = 0.9
weight_entropy = 1

# compute popent score
popent_items_df['popent'] = weight_popularity*np.log10(popent_items_df['freq']) + weight_entropy*popent_items_df['entropy']
popent_items_df.sort_values(by='popent', inplace=True, ascending=False)

print('Computed PopEnt scores for all items')


# Error Score

# function to compute misclassification error
def error(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    error = 1 - probs.max()
    return error


unique_itemId = pd.Series(threshold_item_df)
error_list = np.zeros(shape=(len(unique_itemId), 2))
j = 0

# loop over all itemId's and compute the error for each item
for i in unique_itemId:
    item_i_df = df[df['itemId'] == i]
    error_list[j] = [i, error(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': error_list[:, 0], 'error': error_list[:, 1]}
error_items_df = pd.DataFrame(to_df)
error_items_df.sort_values(by='error', inplace=True, ascending=False)

del error_list

print('Computed error scores for all items')

# PopError Score

# prepare item error for merging
error_items_df2 = error_items_df.sort_values(by='itemId')
error_items_df2.set_index(keys='itemId', inplace=True)

# merge frequencies and errors
poperror_items_df = pd.concat([item_freq_df, error_items_df2], axis=1, join='inner')

# set weights for the poperror score
weight_popularity = 0.9
weight_error = 1

# compute poperror score
poperror_items_df['error'] = weight_popularity*np.log10(poperror_items_df['freq'])+weight_error*poperror_items_df['error']
poperror_items_df.sort_values(by='error', inplace=True, ascending=False)

print('Computed poperror scores for all items')


# Variance Score

# function to compute variance
def variance(labels, users):
    n_labels = len(labels)
    users_u = users.nunique()

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    var = 0.

    for r_ui in labels:
        var += ((r_ui - np.mean(labels))**2)

    variance = var/users_u

    return variance


unique_itemId = pd.Series(threshold_item_df)
variance_list = np.zeros(shape=(len(unique_itemId), 2))
j = 0

# loop over all itemId's and compute the variance for each item
for i in unique_itemId:
    item_i_df = df[(df['itemId'] == i)]
    variance_list[j] = [i, variance(item_i_df['interaction'], item_i_df["userId"])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': variance_list[:, 0], 'variance': variance_list[:, 1]}
variance_items_df = pd.DataFrame(to_df)
variance_items_df.sort_values(by='variance', inplace=True, ascending=False)

del variance_list

print('Computed variance scores for all items')

# PopVar Score

# prepare item variance for merging
variance_items_df2 = variance_items_df.sort_values(by='itemId')
variance_items_df2.set_index(keys='itemId', inplace=True)

# merge frequencies and variances
popvar_items_df = pd.concat([item_freq_df, variance_items_df2], axis=1, join='inner')

# set weights for the popvar score
weight_popularity = 0.9
weight_variance = 1

# compute popvar score
popvar_items_df['variance'] = weight_popularity*np.log10(popvar_items_df['freq'])+weight_variance*popvar_items_df['variance']
popvar_items_df.sort_values(by='variance', inplace=True, ascending=False)

print('Computed popvar scores for all items')

# Prepare the Ranking Strategies

# set the number of items to show to the cold user
nr_of_shown_items = len(threshold_item_df)
print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))

# POPULARITY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) sorted by popularity (purchase/return frequency)
pop_items = item_freq_counter.most_common(nr_of_shown_items)
pop_items = [x[0] for x in pop_items]
pop_items = np.array(pop_items, dtype='int64')
print('Computed ranking using popularity strategy')

# GINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest Gini score
gini_items = gini_items_df.head(nr_of_shown_items)['itemId']
gini_items = np.array(gini_items, dtype=np.int64)
print('Computed ranking using Gini strategy')

# POPGINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popgini score
popgini_items = popgini_items_df.head(nr_of_shown_items)
popgini_items = np.array(popgini_items.index.values, dtype=np.int64)
print('Computed ranking using popgini strategy')

# ENTROPY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest entropy
ent_items = ent_items_df.head(nr_of_shown_items)['itemId']
ent_items = np.array(ent_items, dtype=np.int64)
print('Computed ranking using entropy strategy')

# POPENT STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
popent_items = popent_items_df.head(nr_of_shown_items)
popent_items = np.array(popent_items.index.values, dtype=np.int64)
print('Computed ranking using popent strategy')

# ERROR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest error score
error_items = error_items_df.head(nr_of_shown_items)["itemId"]
error_items = np.array(error_items, dtype=np.int64)
print('Computed ranking using error strategy')

# POPERROR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
poperror_items = poperror_items_df.head(nr_of_shown_items)
poperror_items = np.array(poperror_items.index.values, dtype=np.int64)
print('Computed ranking using poperror strategy')

# VARIANCE STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest variance score
variance_items = variance_items_df.head(nr_of_shown_items)["itemId"]
variance_items = np.array(variance_items, dtype=np.int64)
print('Computed ranking using variance strategy')

# POPVAR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popvar score
popvar_items = popvar_items_df.head(nr_of_shown_items)
popvar_items = np.array(popvar_items.index.values, dtype=np.int64)
print('Computed ranking using popvar strategy')


# RL Agent for all Active Learning Strategies

# Custom MLP policy of three layers of size 64, and 32
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 32],
                                           layer_norm=False,
                                           act_fun=tf.nn.tanh,
                                           feature_extraction="mlp")


env = DummyVecEnv([lambda: rec_env_AL(df,
                                      pop_items,
                                      gini_items,
                                      popgini_items,
                                      ent_items,
                                      popent_items,
                                      error_items,
                                      poperror_items,
                                      variance_items,
                                      popvar_items,
                                      cold_users,
                                      threshold_item_df,
                                      recs=10)])


model = DQN(CustomPolicy,
            env,
            verbose=1,
            gamma=0.99,
            learning_rate=0.0004,
            buffer_size=100,
            exploration_fraction=0.9,
            exploration_final_eps=0.01,
            target_network_update_freq=100,
            train_freq=10,
            batch_size=32,
            learning_starts=50,
            prioritized_replay=True)


model.learn(total_timesteps=2000, log_interval=50)


# RL Agent for the Items

# Custom MLP policy of three layers of size 64, and 32
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           layers=[64, 32],
                                           layer_norm=False,
                                           act_fun=tf.nn.tanh,
                                           feature_extraction="mlp")


env = DummyVecEnv([lambda: rec_env_items(df,
                                         pop_items[:200],
                                         cold_users,
                                         threshold_item_df,
                                         recs=10)])


model = DQN(CustomPolicy,
            env,
            verbose=1,
            gamma=0.99,
            learning_rate=0.0004,
            buffer_size=100,
            exploration_fraction=0.9,
            exploration_final_eps=0.01,
            target_network_update_freq=100,
            train_freq=10,
            batch_size=32,
            learning_starts=50,
            prioritized_replay=True)

model.learn(total_timesteps=2000, log_interval=50)
