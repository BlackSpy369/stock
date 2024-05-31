import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data:np.ndarray,n_length:int,standarize:bool=False):
    if standarize:
        scalar=MinMaxScaler()
        data=scalar.fit_transform(data.reshape(-1,1)).reshape(data.shape[0],)
        
    X=[]
    y=[]
    for i in range(len(data)-n_length):
        X.append(data[i:i+n_length])
        y.append(data[i+n_length])
        
    return np.array(X),np.array(y)
