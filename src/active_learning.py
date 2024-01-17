from __future__ import division
import random
import collections
import csv
import numpy as np
import turicreate as gl
import pandas as pd
import math as math


# load dataset
data = gl.SFrame.read_csv("useritemmatrix.csv", header=True, delimiter=',')
data.remove_column("X1")
# transform dataset from graphlab SFrame to pandas DataFrame
data_pd = gl.SFrame.to_dataframe(data)
data_pd["itemId"] = data_pd["itemId"].astype("category").cat.codes
# TRAINING SET PARTITIONING

# checking users who have many purchases
user_freq_df = pd.DataFrame.from_dict(collections.Counter(data_pd['userId']), orient='index').reset_index()
user_freq_df = user_freq_df.rename(columns={'index': 'userId', 0: 'freq'})

# percentage of total number of users to set as cold user
perc_cold_users = 0.25
nr_of_cold_users = int(math.floor(len(user_freq_df)*perc_cold_users))

# select the [nr_of_cold_users] with the highest number of interactions
cold_users = user_freq_df.sample(nr_of_cold_users, random_state=1)
cold_users = cold_users.get_value(index=range(0, (nr_of_cold_users)), col=0, takeable=True)

print('Selecting ' + str(nr_of_cold_users) + ' cold user(s)')


# SETTINGS FOR SHOWN ITEMS (ranking lengths and item frequency threshold) AND COMPUTING THE GINI, ENTROPY AND POPENT SCORES FOR THE ITEMS


# compute purchase purchase/return frequency per item
item_freq_counter = collections.Counter(data_pd['itemId'])
item_freq_df = pd.DataFrame.from_dict(item_freq_counter, orient='index').reset_index()
item_freq_df = item_freq_df.rename(columns={'index': 'itemId', 0: 'freq'})

# produce list of items which are at least interacted with [threshold_item] times
threshold_item = 10
threshold_item_df = item_freq_df[item_freq_df['freq'] >= threshold_item]['itemId']


# GINI SCORE
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
    item_i_df = data_pd[data_pd['itemId'] == i]
    gini_list[j] = [i, gini(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': gini_list[:, 0], 'gini': gini_list[:, 1]}
gini_items_df = pd.DataFrame(to_df)
gini_items_df.sort_values(by='gini', inplace=True, ascending=False)


del gini_list

print('Computed Gini scores for all items')


#  ERROR SCORE
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
    item_i_df = data_pd[data_pd['itemId'] == i]
    error_list[j] = [i, error(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': error_list[:, 0], 'error': error_list[:, 1]}
error_items_df = pd.DataFrame(to_df)
error_items_df.sort_values(by='error', inplace=True, ascending=False)

del error_list

print('Computed error scores for all items')


# VARIANCE SCORE
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
    item_i_df = data_pd[(data_pd['itemId'] == i)]
    variance_list[j] = [i, variance(item_i_df['interaction'], item_i_df["userId"])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': variance_list[:, 0], 'variance': variance_list[:, 1]}
variance_items_df = pd.DataFrame(to_df)
variance_items_df.sort_values(by='variance', inplace=True, ascending=False)

del variance_list

print('Computed variance scores for all items')


# ENTROPY SCORE
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
    item_i_df = data_pd[data_pd['itemId'] == i]
    entropy_list[j] = [i, entropy(item_i_df['interaction'])]
    j += 1

# transform to pandas DataFrame
to_df = {'itemId': entropy_list[:, 0], 'entropy': entropy_list[:, 1]}
ent_items_df = pd.DataFrame(to_df)
ent_items_df.sort_values(by='entropy', inplace=True, ascending=False)

del entropy_list

print('Computed entropy scores for all items')

# prepare item purchase counts for merging
item_freq_df.sort_values(by='itemId', inplace=True)
item_freq_df.set_index(keys='itemId', inplace=True)
item_freq_df['freq'] = pd.to_numeric(item_freq_df['freq'])

# POPGINI SCORE
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

# POPENT SCORE
# prepare item entropies for merging
ent_items_df2 = ent_items_df.sort_values(by='itemId')
ent_items_df2.set_index(keys='itemId', inplace=True)

# merge frequencies and entropies
popent_items_df = pd.concat([item_freq_df, ent_items_df2], axis=1, join='inner')

# set weights for the popent score
weight_popularity = 0.9
weight_entropy = 1

# compute popent score
popent_items_df['popent'] = weight_popularity*np.log10(popent_items_df['freq'])+weight_entropy*popent_items_df['entropy']
popent_items_df.sort_values(by='popent', inplace=True, ascending=False)

print('Computed PopEnt scores for all items')


# POPENT SCORE WEIGHT OPTIMIZATION

# POPERROR SCORE
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


# POPERROR SCORE WEIGHT OPTIMIZATION

# POPVAR SCORE
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


# POPVAR SCORE WEIGHT OPTIMIZATION

# filename = 'weights_popent_final_new.csv'
# csvfile = open(filename, 'w+')
# writer = csv.writer(csvfile, delimiter=',')
# writer.writerow(['Ranking strategy','Nr. of shown items','Nr. of cold users','RMSE','Weight popularity','Weight entropy'])

# # start k and l for loop here

# weight_pop_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
# weight_ent_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

# nr_of_shown_items_list = [10,100,1000,10000]

# for k in weight_pop_list:
#     for l in weight_ent_list:

#         # set weights for the popent score
#         weight_popularity = k
#         weight_entropy = l
#         # compute popent score
#         popent_items_df['popent'] = weight_popularity*np.log10(popent_items_df['freq'])+weight_entropy*popent_items_df['entropy']
#         popent_items_df.sort_values(by='popent',inplace=True,ascending=False)

#         print('Computed popent scores for all items')
#         print(k)
#         print(l)

# # start m for loop here

#         for m in nr_of_shown_items_list:
#             # set the number of items to show to the cold user
#             nr_of_shown_items = m
#             print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))

#             # POPENT STRATEGY
#             # select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
#             popent_items = popent_items_df.head(nr_of_shown_items)
#             popent_items_final = np.array(popent_items.index.values, dtype=np.int64)
#             print('Computed ranking using popent strategy')

#             # hyperparameter ranges
#             # optimal hyperparameters
#             # num_factors
#             i = 200
#             # regularization
#             j = '1e-06'
#             # linear_regularization
#             h = '1e-07'
#             # number of shown items
#             number_of_shown_items = str(nr_of_shown_items)

#             print('Computing results')

#             # POPENT STRATEGY
#             ranking_strategy = 'PopEnt strategy'
#             # model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
#             train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popent_items_final))]
#             # cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
#             cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popent_items_final))]['userId'])
#             test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popent_items_final))]
#             train = gl.SFrame(train_pd)
#             test = gl.SFrame(test_pd)

#             model = gl.factorization_recommender.create(train,user_id='userId',item_id='itemId',target='interaction',num_factors=i,regularization=j,linear_regularization=h,binary_target=True,max_iterations=50,random_seed=1,verbose=False)

#             print('Rec sys built for popent strategy')

#             rmse = model.evaluate_rmse(test,target='interaction')["rmse_overall"]

#             print('RMSE computed for popent strategy')

#             writer.writerow([ranking_strategy,number_of_shown_items,nr_of_cold_users,rmse,k,l])

#             print('Finished computing results')

# csvfile.close()

# set the number of items to show to the cold user
# !!! in final version, construct a loop here (and test for different number of items shown to the cold user(s))

nr_of_shown_items = 10
print('Number of items shown to the cold user(s): ' + str(nr_of_shown_items))

# RANDOM STRATEGY
# select [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) at random
random_items = random.sample(list(threshold_item_df), nr_of_shown_items)
random_items = np.array(random_items, dtype='int64')

print('Computed ranking using random strategy')

# POPULARITY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) sorted by popularity (purchase/return frequency)
pop_items = item_freq_counter.most_common(nr_of_shown_items)
pop_items = [x[0] for x in pop_items]
pop_items = np.array(pop_items, dtype='int64')
print('Computed ranking using popularity strategy')

# GINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest Gini
gini_items = gini_items_df.head(nr_of_shown_items)['itemId']
gini_items = np.array(gini_items, dtype=np.int64)
print('Computed ranking using Gini strategy')

# ENTROPY STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest entropy
ent_items = ent_items_df.head(nr_of_shown_items)['itemId']
ent_items = np.array(ent_items, dtype=np.int64)
print('Computed ranking using entropy strategy')

# POPGINI STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popgini score
popgini_items = popgini_items_df.head(nr_of_shown_items)
popgini_items = np.array(popgini_items.index.values, dtype=np.int64)
print('Computed ranking using popgini strategy')

# POPENT STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
popent_items = popent_items_df.head(nr_of_shown_items)
popent_items = np.array(popent_items.index.values, dtype=np.int64)
print('Computed ranking using popent strategy')


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# ERROR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest error score
error_items = error_items_df.head(nr_of_shown_items)["itemId"]
error_items = np.array(error_items, dtype=np.int64)
print('Computed ranking using error strategy')


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# VARIANCE STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest variance score
variance_items = variance_items_df.head(nr_of_shown_items)["itemId"]
variance_items = np.array(variance_items, dtype=np.int64)
print('Computed ranking using variance strategy')


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# POPERROR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
poperror_items = poperror_items_df.head(nr_of_shown_items)
poperror_items = np.array(poperror_items.index.values, dtype=np.int64)
print('Computed ranking using poperror strategy')


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# POPVAR STRATEGY
# select the top [nr_of_shown_items] items (which are interacted with at least [threshold_item] times) with largest popent score
popvar_items = popvar_items_df.head(nr_of_shown_items)
popvar_items = np.array(popvar_items.index.values, dtype=np.int64)
print('Computed ranking using popvar strategy')


# COMPUTING THE RESULTS FOR EACH RANKING STRATEGY

# hyperparameter ranges
# optimal hyperparameters
# num_factors
i = 200
# regularization
j = '1e-06'
# linear_regularization
h = '1e-07'
# number of shown items
number_of_shown_items = str(nr_of_shown_items)

print('Computing results')

# RANDOM STRATEGY
ranking_strategy = 'Random strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(random_items))]

# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(random_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(random_items))]

train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for random strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for random strategy')

filename = str('final_results_shown_items_' + number_of_shown_items + '.csv')
csvfile = open(filename, 'w+')
writer = csv.writer(csvfile, delimiter=',')
writer.writerow(['Ranking strategy', 'Nr. of shown items', 'Nr. of cold users', 'RMSE'])
writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])

# POPULARITY STRATEGY
ranking_strategy = 'Popularity strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(pop_items))]

# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(pop_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(pop_items))]

train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for popularity strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for popularity strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# GINI STRATEGY
ranking_strategy = 'Gini strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(gini_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(gini_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(gini_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for Gini strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for Gini strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# ENTROPY STRATEGY
ranking_strategy = 'Entropy strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(ent_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(ent_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(ent_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for entropy strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for entropy strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# POPGINI STRATEGY
ranking_strategy = 'PopGini strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popgini_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popgini_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popgini_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for popgini strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for popgini strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# POPENT STRATEGY
ranking_strategy = 'PopEnt strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popent_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popent_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popent_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for popent strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for popent strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# ERROR STRATEGY
ranking_strategy = 'error strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(error_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(error_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(error_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for error strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for error strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])

# VARIANCE STRATEGY
ranking_strategy = 'variance strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(variance_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(variance_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(variance_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for variance strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for variance strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])


# POPERROR STRATEGY
ranking_strategy = 'poperror strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(poperror_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(poperror_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(poperror_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for poperror strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for poperror strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])

# POPVAR STRATEGY
ranking_strategy = 'popvar strategy'
# model is trained on all user item pairs of the warm users and the user item pairs of the cold user with the shown items (if the cold user has interacted with the shown items)
train_pd = data_pd[(~data_pd.userId.isin(cold_users)) | (data_pd.itemId.isin(popvar_items))]
# cold user(s) that have interacted with one or more of the shown items (if a cold user has not interacted with any of the shown items, it is not included in the test set)
cold_users_interacted = np.array(data_pd[(data_pd.userId.isin(cold_users)) & (data_pd.itemId.isin(popvar_items))]['userId'])
test_pd = data_pd[(data_pd.userId.isin(cold_users_interacted)) & (~data_pd.itemId.isin(popvar_items))]
train = gl.SFrame(train_pd)
test = gl.SFrame(test_pd)

model = gl.factorization_recommender.create(train, user_id='userId', item_id='itemId', target='interaction', num_factors=i, regularization=j, linear_regularization=h, binary_target=True, max_iterations=50, random_seed=1, verbose=False)

print('Rec sys built for popvar strategy')

rmse = model.evaluate_rmse(test, target='interaction')["rmse_overall"]

print('RMSE computed for popvar strategy')

writer.writerow([ranking_strategy, number_of_shown_items, nr_of_cold_users, rmse])

csvfile.close()
