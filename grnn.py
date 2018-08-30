import numpy as np
import pandas as pd

class GRNN :

    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test

        self.std     = np.ones((1,self.y_train.size))#np.random.rand(1,self.train_y.size) #Standard deviations(std) are sometimes called RBF widths.

    def activation_func(self,distances):
        
        return np.exp(- (distances**2) / 2*(self.std**2) )

    def weight(self,i):

        distances=np.sum((self.x_test[i]-self.x_train)**2,axis=1)

        return self.activation_func(distances)
   
    def denominator(self,i):

        return np.sum(self.weight(i))

    def numerator(self,i): 

        return  np.sum(self.weight(i) * self.y_train)
    
    def predict(self):

        predict_array = np.array([])

        for i in range(self.y_test.size):
            predict=np.array([self.numerator(i)/self.denominator(i)])
            predict_array=np.append(predict_array,predict)
        
        return predict_array
    
    def mean_squared_error(self):

        return (self.predict()-self.y_test)**2 /self.y_test.size
    
    def root_mean_squared_error(self):

        return np.sqrt(self.mean_squared_error())
    
train_x = np.array([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
train_y = np.array([4,6,8,10]).T
test_x= np.array([[4,5,6],[2,3,4]])
test_y =np.array([7,5]).T

model=GRNN(train_x,train_y,test_x,test_y)
print('Sonuç :' ,model.predict())
print('Std : ',model.std)
print('Ağırlık(Birinci elamanın) :',model.weight(0))
print('Ağırlık(İkinci elamanın) :',model.weight(1))
print('MSE : ',model.mean_squared_error())
'''
C:\Users\Engin Bozaba>python "c:/Users/Engin Bozaba/Desktop/vs.py"
Sonuç : [7. 5.]
Std :  [[1. 1. 1. 1.]]
Ağırlık(Birinci elamanın) : [[5.00796571e-159 1.11089965e-002 1.11089965e-002 5.00796571e-159]]
Ağırlık(İkinci elamanın) : [[1.11089965e-002 1.11089965e-002 5.00796571e-159 0.00000000e+000]]
MSE :  [0.00000000e+00 3.94430453e-31]
'''
