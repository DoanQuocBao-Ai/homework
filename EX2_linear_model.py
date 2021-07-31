import numpy as np
from matplotlib import pyplot as plt

X_data=[1,2,3,4,5,6]
Y_data=[3,12,27,48,75,108]
w_list = []
mse_list = []
w = np.random.random(1)
def forward(x):
    return x**2*w
def loss(x,y):
    Y=forward(x)
    return (Y-y)*(Y-y)

for w in np.arange(10):
    print("w={}",w)
    dem=0
    for x_val,y_val in zip(X_data,Y_data):
        X_forward = forward(x_val)
        val = loss(x_val, y_val)
        dem+= val
        print("\t",dem , val, X_forward)
    print("MSE = ",dem/3)
    w_list.append(w)
    mse_list.append(dem/3)

plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("MSE")
plt.show()



