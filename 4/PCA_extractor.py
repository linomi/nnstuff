import numpy as np 
from matplotlib import pyplot as plt 

class PCA():
    def __init__(self,input_matrix,input_feature_size,PC_size):
        self.w = np.ones((input_feature_size,PC_size))
        self.x = input_matrix
        self.y = np.matmul(self.x,self.w)
        self.xbar = np.matmul(self.y,np.transpose(self.w))
    def update(self,alpha):
        self.e = self.x - self.xbar 
        # learning rule based on paper
        self.w = self.w + alpha*np.transpose(np.matmul(np.transpose(self.y),self.e))
        self.y = np.matmul(self.x,self.w)
        self.xbar = np.matmul(self.y,np.transpose(self.w))
    def get_pcs(self):
        return self.y 




