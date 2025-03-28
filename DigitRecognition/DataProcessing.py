import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def split_data(X, T, split_ratio = 0.8):
    n_train = int(split_ratio * len(X))
    Xtrain = X[:n_train, ...]
    Ttrain = T[:n_train, ...]
    Xvalid = X[n_train:, ...]
    Tvalid = T[n_train:, ...]
    return (Xtrain, Ttrain), (Xvalid, Tvalid)

def to_gray(X):
    gray = np.dot(X[..., :], [0.2989, 0.5870, 0.1140]) if X.ndim == 3 else X 
    gray = np.expand_dims(gray, -1) if gray.ndim == 2 else gray
    return gray

def resize(Xpath, size=(32, 32)):
    X = Image.open(Xpath)
    X_resized = X.resize(size, Image.LANCZOS)
    return np.array(X_resized)

