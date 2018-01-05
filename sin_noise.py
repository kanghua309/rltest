import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


base=1000
n=2000
noise=0.05

w1 = 150./n

x=np.arange(0,n)
# y=np.sin(x)

y = np.empty(n)

ret = np.sin(x*w1)*0.3 + base*np.clip(np.random.normal(scale=noise,size=(n,)),-0.02,0.02)
#ret = base*np.clip(np.random.normal(scale=noise,size=(n,)),-0.02,0.02)

print(np.clip(np.random.normal(scale=noise,size=(n,)),-0.02,0.02))

y = ret

#print(np.shape(x),np.shape(y))
y[0] = base
for i in range(1,n):
   y[i] = y[i-1] + ret[i]


plt.plot(x,y)
print("---------")
plt.show()
print("--------------")
