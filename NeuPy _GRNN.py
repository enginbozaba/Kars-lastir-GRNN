import numpy as np
from neupy import algorithms, estimators, environment
grnn = algorithms.GRNN(std=1, verbose=False)#0.11

train_X = np.array([[0,1],[0,0],[1,1]])
train_y = np.array([1,0,0]).T
test_X= np.array([[1,0]])
test_y =np.array([1]).T

#eğitim
grnn.train(train_X, train_y)
y_pred = grnn.predict(test_X)

print(y_pred)

# Çıktı :[[0.1553624]]
