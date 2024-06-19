import json

import numpy
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import joblib
pd.options.mode.chained_assignment = None

# create a data frame of this dataset
dataset = pd.read_csv('../data_processing/ratings.dat',
                      sep='::',
                      names=['user', 'item', 'rating', 'timestamp'],
                      dtype={'user': 'int32', 'item': 'int32', 'rating': 'float32'},
                      engine='python').drop('timestamp', axis=1)
# unique dataframe of mean and rating count for each movie
avg_movie_rating = dataset.groupby('item')['rating'].agg(['mean', 'count']).sort_values('count', ascending=False)
# print(avg_movie_rating.tail())
missing_index_dataset = dataset.merge(avg_movie_rating, on='item').sort_values('item')
max_item = max(avg_movie_rating.index)
# insert missing ids of item since K-neighbor cant detect those
all_items_df = pd.DataFrame({'item': range(1, max_item + 1)})
all_users_df = pd.DataFrame({'user': dataset['user'].unique()})
full_dataset = missing_index_dataset[missing_index_dataset['count'] >= 5]
# print(full_dataset.head())
full_dataset['rating'] = full_dataset['rating'] - full_dataset['mean']
full_dataset.drop(['count', 'mean'], axis=1, inplace=True)
train, test = train_test_split(full_dataset, test_size=0.2, random_state=103, shuffle=True)
train_df = pd.merge(all_items_df, train, on='item', how='outer').merge(all_users_df, on='user', how='outer').astype(float).fillna(0)
train_df = train_df._append({'item': 0.0, 'user': 0.0, 'rating': 0.0}, ignore_index=True)
test_df = pd.merge(all_items_df, test, on='item', how='outer').merge(all_users_df, on='user', how='outer').fillna(0)

# ATTENTION!: when run the code for the 1st time, uncomment this block of code to create a
# copy of knn model, which will save your times when running the code several times
# K-nearest-neighbors
model_knn = joblib.load('../data_processing/model_knn.pkl')
# comment and then uncomment this line after the 1st time run the code

test_matrix = test_df.pivot(index='item', columns='user', values='rating').fillna(0).values
train_matrix = train_df.pivot(index='item', columns='user', values='rating').fillna(0).values
# model_knn = NearestNeighbors(n_neighbors=15, metric='cosine')
# model_knn.fit(train_matrix)
# joblib.dump(model_knn, 'model_knn.pkl')


def print_similar_movies(item_input):
    query_index_movie_ratings = test_matrix[item_input, :].reshape(1, -1)
    distances, indices = model_knn.kneighbors(query_index_movie_ratings, n_neighbors=11)
    all_avg_item_rating = all_item_cossim = movie_counter = 0

    for i in range(0, len(distances.flatten())):
        indices_flat = indices.flatten()[i]
        if i == 0 or indices_flat not in avg_movie_rating.index:
            continue
            # print('Recommendations for {0}:\n'.format(item_input))
        elif movie_counter == 10:
            break
        else:
            avg_rating_each_movie = avg_movie_rating.loc[indices_flat, 'mean']
            # check the cosim
            # A = train_matrix[item]
            # B = train_matrix[indices_flat]
            # cosine = np.dot(A.T, B) / (norm(A) * norm(B))
            # print(cosine + distances[0, movie_counter + 1])
            movie_counter += 1
            # print('{0}: {1} with avg rating {2}'.format(i, indices_flat, avg_rating_each_movie))
            all_avg_item_rating += avg_rating_each_movie * (1 - distances.flatten()[movie_counter])
            all_item_cossim += abs(1 - distances.flatten()[movie_counter])

    # print(all_avg_item_rating, all_item_cossim)
    # print('Average rating for {0}: {1}'.format(item_input, abs(all_avg_item_rating) / all_item_cossim))
    # print('True rating for {0}: {1}'.format(item_input, avg_movie_rating.loc[item_input, 'mean']))
    return np.array([item_input, all_avg_item_rating / all_item_cossim])


predict_rating = np.empty((0, 2))
for i in test['item'].unique():
    predict_rating = np.append(predict_rating, [print_similar_movies(i)], axis=0)

np.savetxt('../data_processing/predict_movie_rating.dat', predict_rating, delimiter='::', fmt=['%d', '%.2f'])
