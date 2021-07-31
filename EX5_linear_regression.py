from torch.autograd import Variable
import torch
x_data=Variable(torch.Tensor([[1],[2],[3]]))
y_data=Variable(torch.Tensor([[2],[4],[6]]))

class data(torch.nn.Module):
    def __init__(selt):
        super(data,selt).__init__()
        selt.linear=torch.nn.Linear(1,1)
    def forward(selt,x):
        y_pred=selt.linear(x)
        return y_pred
data_pred=data()

criter=torch.nn.MSELoss(reduction='sum')
optim=torch.optim.ASGD(data_pred.parameters(),lr=0.01)
print("before tranning",4,data_pred.forward(Variable(torch.Tensor([[4]]))).item())
for i in range(1000):
    y_pred=data_pred(x_data)
    loss=criter(y_pred,y_data)
    print(i,"loss",loss.item())
    optim.zero_grad()
    loss.backward()
    optim.step()
print("after tranning",4,data_pred.forward(Variable(torch.Tensor([[4]]))).item())