import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

dataset1 = pd.read_csv('avg_movie_rating.dat', engine='python', header=None)
dataset2 = pd.read_csv('predict_movie_rating.dat', names=[0, 1], delimiter="::", engine='python').drop(columns=0)

dataset1_array = np.array(dataset1)
dataset2_array = np.array(dataset2)

print(dataset2_array)
print(dataset1_array)
# print(dataset2.shape)

rmse = mean_squared_error(dataset1_array[:,0], dataset2_array[:,0], squared=False)
print(rmse)
