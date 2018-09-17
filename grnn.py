import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

class GRNN :

    def __init__(self,x_train,y_train,x_test,y_test):

        self.x_train= x_train
        self.y_train= y_train
        self.x_test= x_test
        self.y_test= y_test

        self.std     = np.full((1,self.y_train.size),1)#np.random.rand(1,self.train_y.size) #Standard deviations(std) are sometimes called RBF widths.

    def activation_func(self,distances):
        
        
        return np.exp(- (distances**2) / 2*(self.std**2) )

    def output(self,i):#sometimes called weight

        distances=np.sum((self.x_train-self.x_test[i])**2,axis=1)**(1/2)

        return self.activation_func(distances)
   
    def denominator(self,i):

        return np.sum(self.output(i))

    def numerator(self,i): 

        return  np.sum(self.output(i) * self.y_train)
    
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
    
    
train_X = np.array([[0,1],[0,0],[1,1]])
train_y = np.array([1,0,0]).T
test_X= np.array([[1,0]])
test_y =np.array([1]).T

model=GRNN(train_X,train_y,test_X,test_y)
print('Sonuç :' ,model.predict())
print('Std : ',model.std)
print('Gizli Katmanın Çıktıları(Birinci elamanın) :',model.output(0))
#print('Gizli Katmanın Çıktıları(İkinci elamanın) :',model.output(1))
print('MSE : ',model.mean_squared_error())

'''
Sonuç : [0.23269654]
Std :  [[1 1 1]]
Gizli Katmanın Çıktıları(Birinci elamanın) : [[0.36787944 0.60653066 0.60653066]]
MSE :  [0.5887546]
'''
