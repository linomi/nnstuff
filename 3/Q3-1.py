#importing stuff
from matplotlib import pyplot as plt 
import math 

#creating input and out put 
N = range(0,100)
y = [(-1)**n*math.log(2**(2*n+1)+0.0001) for n in N]
plt.subplot(1,2,1)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.bar(N,y)
plt.legend(["y[n]"])
plt.subplot(1,2,2)
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.xlim((-1,1))
plt.ylim((-2,2))
plt.arrow(0,0,0,1)
plt.legend(["x[n]"])
plt.show()

