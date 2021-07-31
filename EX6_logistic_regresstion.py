import torch
from torch.autograd import Variable
from torch.nn import functional as F
x_data=Variable(torch.Tensor([[1],[2],[3],[4]]))
y_data=Variable(torch.Tensor([[0],[0],[1],[1]]))
class Modul(torch.nn.Module):
    def __init__(self):
        super(Modul,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=F.tanh(self.linear(x))
        return y_pred
modul=Modul()
criterize=torch.nn.MSELoss(size_average=True)
optimazer=torch.optim.SGD(modul.parameters(),lr=0.01)
for i in range(1000):
    y_pred=modul(x_data)
    loss=criterize(y_pred,y_data)
    print(i,"loss",loss.item())
   
    optimazer.zero_grad()
    loss.backward()
    optimazer.step()
ou=Variable(torch.Tensor([[2.0]]))
print("after trainning:",2.0,modul(ou).item()>0.5)
ou=Variable(torch.Tensor([[9.0]]))
print("after trainning:",9.0,modul(ou).item()>0.5)