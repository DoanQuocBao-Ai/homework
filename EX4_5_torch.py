import torch
from torch.autograd import Variable

A_data = [1,2,3]
B_data = [4,9,16]
b = 1
w1 = Variable(torch.Tensor([1.0]),  requires_grad=True)
w2 = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x**2*w2 + x*w1 + b
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)
print("predic(before trainning)",4,forward(4).item())
for epoch in range(1000):
    for x_val,y_val in zip(A_data,B_data):
        l = loss(x_val,y_val)
        l.backward()
        print(x_val, y_val, w1.grad,w2.grad)
        w1.data -= 0.01* w1.grad.data
        w2.data -= 0.01* w2.grad.data
        w1.grad.zero_()
        w2.grad.zero_()
    print("epoch: ",epoch,"loss",l.item(),w1.data,w2.data)

print("predic(after trainning)",4,forward(4).item())