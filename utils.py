import numpy as np


def preprocess_data(data:np.ndarray,n_length:int):
    X=[]
    y=[]
    for i in range(len(data)-n_length):
        X.append(data[i:i+n_length])
        y.append(data[i+n_length])
        
    return np.array(X),np.array(y)