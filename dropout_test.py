# https://www.zhihu.com/question/66782101
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(x)


class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()

    def forward(self, x):
        return F.dropout(x)


m1 = Model1()
m2 = Model2()
inputs = torch.rand(10)
print(inputs)
print(torch.sum(inputs))
print(20 * '-' + "train model:" + 20 * '-' + '\r\n')
print(m1(inputs))
print(torch.sum(m1(inputs)))
print(m2(inputs))
print(torch.sum(m2(inputs)))
print(20 * '-' + "eval model:" + 20 * '-' + '\r\n')
m1.eval()
m2.eval()
print(m1(inputs))
print(torch.sum(m1(inputs)))
print(m2(inputs))
print(torch.sum(m2(inputs)))
