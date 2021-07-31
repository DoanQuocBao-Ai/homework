import numpy as np
from matplotlib import pyplot as plt

x_array = [1,2,3]
y_array = [4,9,16]
w1 = np.random.random(1)
w2 = np.random.random(1)
b = np.random.random(1)
print("b=",b)
trainning = 0.01
def forward(x):
    return x**2*w2 + x*w1 + b

def gradient(x,y):
    return 2*x*(x**2*w2+x*w1+b-y)
def gradient2(x,y):
    return 2*x**2*(x**2*w2+x*w1+b-y)
def loss(x,y):
    return (x**2*w2+x*w1+b-y)*(x**2*w2+x*w1+b-y)
print("Before trainning",4,forward(4))
for epoch in range(500):
    print("Epoch:",epoch)
    for x_val,y_val in zip(x_array,y_array):
        x_gra= gradient(x_val,y_val)
        w1=w1- trainning* x_gra
        x2_gra= gradient2(x_val,y_val)
        w2=w2- trainning* x2_gra
        l=loss(x_val,y_val)
        print("W1=",w1,"w2=",w2,"loss=",l)
print("After trainning",4,forward(4))