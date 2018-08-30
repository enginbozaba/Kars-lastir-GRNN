import numpy as np
import pandas as pd

class GRNN :

    def __init__(self,train_x,train_y,test_x,test_y):

        self.train_x= train_x
        self.train_y= train_y
        self.test_x= test_x
        self.test_y= test_y

        self.std     = np.ones((1,self.train_y.size))#np.random.rand(1,self.train_y.size) #Standard deviations(std) are sometimes called RBF widths.

    def activation_func(self,distances):
        return np.exp(- (distances**2) / 2*(self.std**2) )

    def weight(self):
        distances=np.sum((self.test_x-self.train_x)**2,axis=1)
        return self.activation_func(distances)
   
    def denominator(self):
        return np.sum(self.weight())

    def numerator(self):      
        return  np.sum(self.weight() * self.train_y)
    
    def predict(self):
        return self.numerator()/self.denominator()
    
    def mean_squared_error(self):
        return (self.predict()-self.test_y)**2 /self.test_y.size
    
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
