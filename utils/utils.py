import numpy as np
from sklearn.preprocessing import MinMaxScaler

scalar=MinMaxScaler()

def preprocess_data(data:np.ndarray,n_length:int,standarize:bool=False):
    if standarize:
        data=scalar.fit_transform(data.reshape(-1,1)).reshape(data.shape[0],)
        
    X=[]
    y=[]
    for i in range(len(data)-n_length):
        X.append(data[i:i+n_length])
        y.append(data[i+n_length])
        
    return np.array(X),np.array(y)


def predict_stock(X:np.ndarray,model,n:int,standarize:bool=False):
    X=X.copy()

    for i in range(n):
        if standarize:
            X=np.append(X,scalar.inverse_transform(model.predict(np.expand_dims(X,axis=0)).reshape(-1,1))[0])
        else:
            X=np.append(X,model.predict(np.expand_dims(X,axis=0)))
        # print(X)
        X=np.delete(X,0)
        # print(f"-- {X}")

    return X