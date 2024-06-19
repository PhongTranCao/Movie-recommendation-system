import numpy as np
import pandas as pd
import pickle
from FunkSVD.svd import SVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_preprocess import ratings_to_df, movies_to_df, convert_douban_to_df
from itertools import product

# load dataframe (ratings and movies) (ML-1M)
df = ratings_to_df("dataset/ratings.dat") 
movies_df = movies_to_df("dataset/movies.dat")

# train test split (ML-1M)
train = df.sample(frac=0.9)
val = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
test = df.drop(train.index.tolist()).drop(val.index.tolist())

######
# load dataframe (ratings) (Douban)
douban_training_df, douban_test_df = convert_douban_to_df()
douban_val_df = douban_training_df.sample(frac=0.2, random_state=8)

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
                early_stopping=True, shuffle=False,
                min_rating=1, max_rating=5)
        
        # ML-1M
        # svd.fit(X=train, X_val=val)
        
        # Douban
        svd.fit(X=douban_training_df, X_val=douban_val_df)
        pred = svd.predict(douban_test_df)

        #pred = svd.predict(test)
        score = np.sqrt(mean_squared_error(douban_test_df["rating"], pred))

        if score < best_score:
            best_score = score
            best_params = params
            
    return best_params, best_score
    
def train_save_model():
    # using hyperparameters from random_search()
    lr, reg, factors = [0.009976, 0.008510, 100]
    
    svd = SVD(lr=lr, reg=reg, n_epochs=50, n_factors=factors,
                early_stopping=True, shuffle=False,
                min_rating=1, max_rating=5)
        
    svd.fit(X=douban_training_df, X_val=douban_val_df)

    # For test set
    pred = svd.predict(douban_test_df)
    mae = mean_absolute_error(douban_test_df["rating"], pred)
    rmse = np.sqrt(mean_squared_error(douban_test_df["rating"], pred))
    
    # For training set
    #pred_train = svd.predict(train)
    #mae_train = mean_absolute_error(train["rating"], pred_train)
    #rmse_train = np.sqrt(mean_squared_error(train["rating"], pred_train))
    
    #print("Training RMSE: {:.4f}".format(rmse_train))
    #print("Training MAE:  {:.4f}".format(mae_train))
    print("Test RMSE: {:.4f}".format(rmse))
    print("Test MAE:  {:.4f}".format(mae))
    print('{} factors, {} lr, {} reg'.format(factors, lr, reg))
    
    # save model
    #with open('loaded_model/funksvd.pkl', 'wb') as file:
    #    pickle.dump(svd, file)
        
def make_prediction(userID, df):
    #Adding our own ratings
    n_m = len(df.i_id.unique())

    #  Initialize my ratings
    my_ratings = np.zeros(n_m)

    my_ratings[3114] = 5

    print('User ratings:')
    print('-----------------')

    for i, val in enumerate(my_ratings):
        if val > 0:
            print('Rated %d stars: %s' % (val, movies_df.loc[movies_df.i_id==i].title.values))
            
    print("Adding your recommendations!")
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
    with open('loaded_model/funksvd.pkl', 'rb') as file:
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
    recommendations.head(10)

    sorted_user_predictions = recommendations.sort_values(by='prediction', ascending=False)

    user_ratings = user_ratings[user_ratings.u_id == userID[0]]
    user_ratings.columns = ['u_id',	'i_id', 'rating']
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = movies_df[~movies_df['i_id'].isin(user_ratings['i_id'])].\
        merge(pd.DataFrame(sorted_user_predictions).reset_index(drop=True), how = 'inner', left_on = 'i_id', right_on = 'i_id').\
        sort_values(by='prediction', ascending = False)#.drop(['i_id'],axis=1)

    rated_df = movies_df[movies_df['i_id'].isin(user_ratings['i_id'])].\
        merge(pd.DataFrame(user_ratings).reset_index(drop=True), how = 'inner', left_on = 'i_id', right_on = 'i_id')
    rated_df = rated_df.loc[rated_df.u_id==userID[0]].sort_values(by='rating', ascending = False)
    
    print(recommendations.head(20))
    
def main():
    
    # Random Search:
    #best_params, best_score = random_search(param_ranges, it=50)
    #print("Best parameters found:", best_params)
    #print("Best RMSE: {:.4f}".format(best_score))
    
    # Train & Save Model
    train_save_model()
    
    #make_prediction(0, df)

if __name__ == "__main__":
    main()
    