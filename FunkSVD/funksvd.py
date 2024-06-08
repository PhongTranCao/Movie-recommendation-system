import json

import numpy as np
import pandas as pd
import pickle
from FunkSVD.svd import SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocess import ratings_to_df, movies_to_df
from itertools import product

# load dataframe (ratings and movies)
df = ratings_to_df("../data_processing/ratings.dat")
# movies_df = movies_to_df("../data_processing/movies.dat")

# train test split
train = df.sample(frac=0.7)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

# generate hyperparameters
def sample_params():
    lr = np.random.uniform(low = 0.001, high = 0.01,  size = 1)[0]
    reg = np.random.uniform(low = 0.001, high = 0.01,  size = 1)[0]
    factors = 50
    return lr, reg, factors

param_ranges = {'lr': (0.001, 0.01), 'reg': (0.001, 0.01)}

def random_search(param_ranges, it):
    best_params = None
    best_score = float('inf')
    for _ in range(it):
        params = {name: np.random.uniform(low, high) for name, (low, high) in param_ranges.items()}
        lr = params['lr']
        reg = params['reg']
        factors = 100
        
        svd = SVD(lr=lr, reg=reg, n_epochs=10, n_factors=factors,
                early_stopping=False, shuffle=False,
                min_rating=1, max_rating=5)
        
        svd.fit(X=train, X_val=val)

        pred = svd.predict(test)
        mae = mean_absolute_error(test["rating"], pred)
        score = np.sqrt(mean_squared_error(test["rating"], pred))
        
        # print("Test RMSE: {:.4f}".format(rmse))
        # print("Test MAE:  {:.4f}".format(mae))
        # print('{} factors, {} lr, {} reg'.format(factors, lr, reg))
        if score < best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score
    
def train_save_model():
    # using hyperparameters from random_search()
    lr, reg, factors = [0.006, 0.009, 100]
    
    svd = SVD(lr=lr, reg=reg, n_epochs=100, n_factors=factors,
                early_stopping=True, shuffle=False,
                min_rating=1, max_rating=5)
        
    svd.fit(X=train, X_val=val)

    # For test set
    pred = svd.predict(test)
    mae = mean_absolute_error(test["rating"], pred)
    rmse = np.sqrt(mean_squared_error(test["rating"], pred))
    
    # For training set
    pred_train = svd.predict(train)
    mae_train = mean_absolute_error(train["rating"], pred_train)
    rmse_train = np.sqrt(mean_squared_error(train["rating"], pred_train))
    
    # print("Training RMSE: {:.4f}".format(rmse_train))
    # print("Training MAE:  {:.4f}".format(mae_train))
    # print("Test RMSE: {:.4f}".format(rmse))
    # print("Test MAE:  {:.4f}".format(mae))
    # print('{} factors, {} lr, {} reg'.format(factors, lr, reg))
    
    # save model
    with open('loaded_model/funksvd.pkl', 'wb') as file:
        pickle.dump(svd, file)
        
def make_prediction(userID, df):
    #Adding our own ratings
    n_m = len(df.i_id.unique())
    prediction_temp = np.empty((n_m, 2))

    #  Initialize my ratings
    my_ratings = np.zeros(n_m)

    my_ratings[112] = 5
    my_ratings[334] = 4
    my_ratings[260] = 5
    my_ratings[1111] = 3
    my_ratings[435] = 5
    my_ratings[1222] = 2
    my_ratings[2628] = 5
    my_ratings[3012] = 1

    # print('User ratings:')
    # print('-----------------')
    #
    # for i, val in enumerate(my_ratings):
    #     if val > 0:
    #         print('Rated %d stars: %s' % (val, movies_df.loc[movies_df.i_id==i].title.values))
            
    # print("Adding your recommendations!")
    items_id = [item[0] for item in np.argwhere(my_ratings>0)]
    ratings_list = my_ratings[np.where(my_ratings>0)]
    user_id = np.asarray([0] * len(ratings_list))
    user_ratings = pd.DataFrame(list(zip(user_id, items_id, ratings_list)), columns=['u_id', 'i_id', 'rating'])
    
    try:
        df = df.drop(columns=['timestamp'])
    except:
        pass
    df = df._append(user_ratings, ignore_index=True)
    
    # Load the pre-trained model and fine-tuning
    with open('funksvd.pkl', 'rb') as file:
        model = pickle.load(file)
        
    train_user = df.sample(frac=0.8)
    val_user = df.drop(train_user.index.tolist()).sample(frac=0.5, random_state=8)
    # test_user = data_with_user.drop(train_user.index.tolist()).drop(val_user.index.tolist())
    model.fit(X=train_user, X_val=val_user)
    
    userID = [userID]
    movies = df.i_id.unique()
    recommendations = pd.DataFrame(list(product(userID, movies)), columns=['u_id', 'i_id'])
    
    # Time to recommend
    pred_train = model.predict(recommendations)
    recommendations['prediction'] = pred_train

    prediction_temp[:, 0] = recommendations['i_id']
    prediction_temp[:, 1] = recommendations['prediction']
    return prediction_temp

    # sorted_user_predictions = recommendations.sort_values(by='prediction', ascending=False)
    #
    # user_ratings = user_ratings[user_ratings.u_id == userID[0]]
    # user_ratings.columns = ['u_id',	'i_id', 'rating']
    # # Recommend the highest predicted rating movies that the user hasn't seen yet.
    # recommendations = movies_df[~movies_df['i_id'].isin(user_ratings['i_id'])].\
    #     merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how = 'inner', left_on = 'i_id', right_on = 'i_id').\
    #     sort_values(by='prediction', ascending = False)#.drop(['i_id'],axis=1)
    #
    # rated_df = movies_df[movies_df['i_id'].isin(user_ratings['i_id'])].\
    #     merge(pd.DataFrame(user_ratings).reset_index(drop=True), how = 'inner', left_on = 'i_id', right_on = 'i_id')
    # rated_df = rated_df.loc[rated_df.u_id==userID[0]].sort_values(by='rating', ascending = False)
    #
    # print(recommendations.head(20))


def print_similar_movies(item_input):
    prediction_temp = make_prediction(item_input, df)
    result = prediction_temp[prediction_temp[:, 0] == item_input]
    if result.size > 0:
        target_prediction = result[0, 1]

        differences = np.abs(prediction_temp[:, 1] - target_prediction)

        closest_indices = np.argsort(differences)[1:11]  # exclude the target itself if present

        closest_i_ids = prediction_temp[closest_indices, 0].astype(int).flatten().tolist()
        closest_i_ids_list = [{"value": "{0}".format(value)} for value in closest_i_ids]
    with open('../data_processing/movies_recommend_id_list.json', 'w') as file:
        json.dump(closest_i_ids_list, file, ensure_ascii=False, indent=4)
    with open('../data_processing/movies_no_tags.json', 'r') as file:
        data1 = json.load(file)
    with open('../data_processing/movies_recommend_id_list.json', 'r') as file:
        data2 = json.load(file)
    lookup = {entry['value']: entry['label'] for entry in data1}
    result = [lookup[entry['value']] for entry in data2 if entry['value'] in lookup]
    result = [{"label": "{0}".format(movie)} for movie in result]
    with open('../data_processing/movies_recommend_name_list.json', 'w') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)

# def main():
#
#     # Random Search:
#     # best_params, best_score = random_search(param_ranges, it=50)
#     # print("Best parameters found:", best_params)
#     # print("Best RMSE: {:.4f}".format(best_score))
#
#     '''
#     # Train & Save Model
#     train_save_model()
#     '''
#
#     print_similar_movies(1)
#
# if __name__ == "__main__":
#     main()
    