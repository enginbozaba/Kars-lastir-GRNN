import numpy as np
import pandas as pd

class GRNN :

    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train= train_x
        self.y_train= train_y
        self.x_test= test_x
        self.y_test= test_y

        self.std     = np.ones((1,self.y_train.size))#np.random.rand(1,self.train_y.size) #Standard deviations(std) are sometimes called RBF widths.

    def activation_func(self,distances):
        return np.exp(- (distances**2) / 2*(self.std**2) )

    def weight(self):
        distances=np.sum((self.x_test-self.x_train)**2,axis=1)
        return self.activation_func(distances)
   
    def denominator(self):
        return np.sum(self.weight())

    def numerator(self):      
        return  np.sum(self.weight() * self.y_train)
    
    def predict(self):
        return self.numerator()/self.denominator()
    
    def mean_squared_error(self):
        return (self.predict()-self.y_test)**2 /self.y_test.size
    
    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())
    
train_x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
train_y = np.array([4,6,8,10]).T
test_x= np.array([[4,5,6]])
test_y =np.array([7])

model=GRNN(train_x,train_y,test_x,test_y)
print('Sonuç :' ,model.predict())
print('Std : ',model.std)
print('Ağırlık :',model.weight())
print('MSE : ',model.mean_squared_error())

print("-----------------------------------------")

train_x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
train_y = np.array([4,6,8,10]).T
test_x= np.array([[2,3,4]])
test_y =np.array([5])

model=GRNN(train_x,train_y,test_x,test_y)
print('Sonuç :' ,model.predict())
print('Std : ',model.std)
print('Ağırlık :',model.weight())
print('MSE : ',model.mean_squared_error())

"""
C:\Users\Engin Bozaba>python "c:/Users/Engin Bozaba/Desktop/vs.py"
Sonuç : 7.0
Std :  [[1. 1. 1. 1.]]
Ağırlık : [[5.00796571e-159 1.11089965e-002 1.11089965e-002 5.00796571e-159]]
MSE :  [0.]
-----------------------------------------
Sonuç : 4.999999999999999
Std :  [[1. 1. 1. 1.]]
Ağırlık : [[1.11089965e-002 1.11089965e-002 5.00796571e-159 0.00000000e+000]]
MSE :  [7.88860905e-31]
"""
