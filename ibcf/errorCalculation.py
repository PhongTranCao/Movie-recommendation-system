import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

avg_movie_train_df = pd.read_csv('../data_processing/avg_movie_rating.dat', engine='python', header=None, names=['true_rating'])
avg_movie_train_df['item'] = avg_movie_train_df.index + 1
avg_movie_test_df = pd.read_csv('../data_processing/avg_movie_rating.dat', engine='python', header=None, names=['true_rating'])
avg_movie_test_df['item'] = avg_movie_test_df.index + 1

predict_df = pd.read_csv('../data_processing/predict_test_movie_rating.dat', delimiter='::', engine='python', header=None, names=['item', 'predicted_rating'])
predict_df.replace({'predicted_rating': np.nan}, 0, inplace=True)

predict_train_df = pd.read_csv('../data_processing/predict_train_movie_rating.dat', delimiter='::', engine='python', header=None, names=['item', 'predicted_rating'])
predict_train_df.replace({'predicted_rating': np.nan}, 0, inplace=True)

merged_test_df = pd.merge(predict_df, avg_movie_test_df, on='item')
rmse_test = np.sqrt(mean_squared_error(merged_test_df['true_rating'], merged_test_df['predicted_rating']))
mae_test = mean_absolute_error(merged_test_df['true_rating'], merged_test_df['predicted_rating'])

merged_train_df = pd.merge(predict_train_df, avg_movie_train_df, on='item')
rmse_train = np.sqrt(mean_squared_error(merged_train_df['true_rating'], merged_train_df['predicted_rating']))
mae_train = mean_absolute_error(merged_train_df['true_rating'], merged_train_df['predicted_rating'])

print("RMSE test:", rmse_test)
print("MAE test:", mae_test)
print("RMSE train:", rmse_train)
print("MAE train:", mae_train)
