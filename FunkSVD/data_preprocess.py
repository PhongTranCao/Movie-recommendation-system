import datetime
import numpy as np
import pandas as pd

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