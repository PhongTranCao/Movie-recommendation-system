import json
import pandas as pd
import joblib


dataset = pd.read_csv('../data_processing/ratings.dat',
                      sep='::',
                      names=['user', 'item', 'rating', 'timestamp'],
                      engine='python').drop('timestamp', axis=1)
print(dataset.head())
avg_movie_rating = pd.DataFrame(dataset.groupby('item')['rating'].agg(['mean', 'count']))
missing_index_dataset = dataset.merge(avg_movie_rating, on='item').sort_values('item')
max_item = max(avg_movie_rating.index)
all_items_df = pd.DataFrame({'item': range(1, max_item + 1)})
full_dataset = pd.merge(all_items_df, missing_index_dataset, on='item', how='left')



model_knn = joblib.load('../data_processing/model_knn.pkl')
movie_wide = full_dataset.pivot(index='item', columns='user', values='rating').fillna(0).values


def print_similar_movies(item_input):
    item = item_input-1
    query_index_movie_ratings = movie_wide[item, :].reshape(1, -1)
    distances, indices = model_knn.kneighbors(query_index_movie_ratings, n_neighbors=11)
    indices_flat = indices.flatten().reshape(-1) + 1
    indices_flat_list = [{"value": "{0}".format(value)} for value in indices_flat]
    with open('../data_processing/movies_recommend_id_list.json', 'w') as file:
        json.dump(indices_flat_list, file, ensure_ascii=False, indent=4)
    with open('../data_processing/movies_no_tags.json', 'r') as file:
        data1 = json.load(file)
    with open('../data_processing/movies_recommend_id_list.json', 'r') as file:
        data2 = json.load(file)
    lookup = {entry['value']: entry['label'] for entry in data1}
    result = [lookup[entry['value']] for entry in data2 if entry['value'] in lookup]
    result = [{"label": "{0}".format(movie)} for movie in result]
    with open('../data_processing/movies_recommend_name_list.json', 'w') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)
