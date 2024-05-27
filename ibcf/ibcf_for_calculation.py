import json

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
import joblib

# create a data frame of this dataset
dataset = pd.read_csv('ratings.dat',
                      sep='::',
                      names=['user', 'item', 'rating', 'timestamp'],
                      engine='python').drop('timestamp', axis=1)
# print(dataset)
# unique dataframe of mean and rating count for each movie
avg_movie_rating = pd.DataFrame(dataset.groupby('item')['rating'].agg(['mean', 'count']))
# print(avg_movie_rating.tail())
# merge dataframes
missing_index_dataset = dataset.merge(avg_movie_rating, on='item').sort_values('item')
max_item = max(avg_movie_rating.index)
# insert missing ids of item since K-neighbor cant detect those
all_items_df = pd.DataFrame({'item': range(1, max_item + 1)})
full_dataset = pd.merge(all_items_df, missing_index_dataset, on='item', how='left')
# print(full_dataset.loc[full_dataset['item'] == 1195])


# ATTENTION!: when run the code for the 1st time, uncomment this block of code to create a
# copy of knn model, which will save your times when running the code several times
# K-nearest-neighbors
model_knn = joblib.load('../data_processing/model_knn.pkl')
# comment and then uncomment this line after the 1st time run the code
movie_wide = full_dataset.pivot(index='item', columns='user', values='rating').fillna(0).values
# print(np.unique(movie_wide[1194]))
# model_knn = NearestNeighbors(n_neighbors=11, metric='cosine')
# model_knn.fit(movie_wide)
# joblib.dump(model_knn, 'model_knn.pkl')


def print_similar_movies(item_input):
    item = item_input-1
    query_index_movie_ratings = movie_wide[item, :].reshape(1, -1)
    # for some reason use Dataframe will cause error and more time in executing
    distances, indices = model_knn.kneighbors(query_index_movie_ratings, n_neighbors=11)
    all_avg_item_rating = all_item_cossim = movie_counter = 0

    for i in range(0, len(distances.flatten())):
        indices_flat = indices.flatten()[i] + 1
        # np.savetxt("indices.dat", indices_flat, fmt='%d')
        if i == 0:
            continue
            # print('Recommendations for {0}:\n'.format(item_input))
        else:
            avg_rating_each_movie = avg_movie_rating.loc[indices_flat, 'mean']
            # check the cosim
            # A = movie_wide[item]
            # B = movie_wide[indices_flat]
            # cosine = np.dot(A.T, B) / (norm(A) * norm(B))
            # print(cosine + distances[0, movie_counter + 1])
            movie_counter += 1
            # print('{0}: {1} with avg rating {2}'.format(i, indices_flat, avg_rating_each_movie))
            all_avg_item_rating += avg_rating_each_movie * (1 - distances.flatten()[movie_counter])
            all_item_cossim += abs(1 - distances.flatten()[movie_counter])

    print(all_avg_item_rating, all_item_cossim)
    print('Average rating for {0}: {1}'.format(item_input, abs(all_avg_item_rating) / all_item_cossim))
    print('True rating for {0}: {1}'.format(item_input, avg_movie_rating.loc[item_input, 'mean']))
    return np.array([item_input, all_avg_item_rating / all_item_cossim])


predict_rating = np.empty((0, 2))
for i in avg_movie_rating.index:
    predict_rating = np.append(predict_rating, [print_similar_movies(i)], axis=0)

np.savetxt('predict_movie_rating.dat', predict_rating, delimiter='::', fmt=['%d', '%.2f'])