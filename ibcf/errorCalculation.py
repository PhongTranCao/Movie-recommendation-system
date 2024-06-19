import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

avg_movie_df = pd.read_csv('../data_processing/avg_movie_rating.dat', engine='python', header=None, names=['true_rating'])
avg_movie_df['item'] = avg_movie_df.index + 1
predict_df = pd.read_csv('../data_processing/predict_movie_rating.dat', delimiter='::', engine='python', header=None, names=['item', 'predicted_rating'])

predict_df.replace({'predicted_rating': np.nan}, 0, inplace=True)

merged_df = pd.merge(predict_df, avg_movie_df, on='item')

rmse = np.sqrt(mean_squared_error(merged_df['true_rating'], merged_df['predicted_rating']))
mae = mean_absolute_error(merged_df['true_rating'], merged_df['predicted_rating'])

print("RMSE:", rmse)
print("MAE:", mae)
