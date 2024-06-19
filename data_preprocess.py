import datetime
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csc_matrix

def ratings_to_df (path):
        names = ['u_id', 'i_id', 'rating', 'timestamp']
        dtype = {'u_id': np.uint32, 'i_id': np.uint32, 'rating': np.float64}

        def date_parser(time):
                return datetime.datetime.fromtimestamp(float(time))

        df = pd.read_csv(path, 
                        names=names, 
                        dtype=dtype, 
                        header=0,
                        sep=r'::',
                        # date_parser=date_parser,
                        engine='python')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df.sort_values(by='timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

def movies_to_df (path):
        names = ['i_id', 'title', 'genres']
        
        df = pd.read_csv(path, names = names, sep=r'::', encoding='latin-1', engine='python')
        df.drop([0], inplace=True)
        df['i_id'] = df['i_id'].apply(pd.to_numeric)
        
        return df

def load_matlab_file(path_file, name_field):
    
    db = h5py.File(path_file, 'r')
    ds = db[name_field]

    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out

def load_data_monti(path='dataset/'):

    M = load_matlab_file(path+'douban.mat', 'M')
    Otraining = load_matlab_file(path+'douban.mat', 'Otraining') * M
    Otest = load_matlab_file(path+'douban.mat', 'Otest') * M

    n_u = M.shape[0]  # num of users
    n_m = M.shape[1]  # num of movies
    n_train = Otraining[np.where(Otraining)].size  # num of training ratings
    n_test = Otest[np.where(Otest)].size  # num of test ratings

    train_r = Otraining.T
    test_r = Otest.T

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    #print('data matrix loaded')
    #print('num of users: {}'.format(n_u))
    #print('num of movies: {}'.format(n_m))
    #print('num of training ratings: {}'.format(n_train))
    #print('num of test ratings: {}'.format(n_test))

    return n_m, n_u, train_r, train_m, test_r, test_m

def convert_douban_to_df ():
    # Load data
    n_m, n_u, train_r, train_m, test_r, test_m = load_data_monti()

    train_r_indices = np.where(train_r > 1e-12)
    train_r_df = pd.DataFrame({
        'u_id': train_r_indices[0],
        'i_id': train_r_indices[1],
        'rating': train_r[train_r_indices]
    })

    test_r_indices = np.where(test_r > 1e-12)
    test_r_df = pd.DataFrame({
        'u_id': test_r_indices[0],
        'i_id': test_r_indices[1],
        'rating': test_r[test_r_indices]
    })
    
    #print("Training Ratings DataFrame:")
    #print(train_r_df)
    #print("Test Ratings DataFrame:")
    #print(test_r_df)
    
    return train_r_df, test_r_df

#convert_douban_to_df()