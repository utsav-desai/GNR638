import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        print(f'O1 :{x}\n')
        x = self.fc2(x)
        print(f'O2 :{x}\n')
        return x


model = SimpleModel()

sample_input = torch.tensor([[1.0, 1.0]], requires_grad=True)

output = model(sample_input)
# print(f'Output: {output}')

target = torch.tensor([[0.0, 0.0]])

loss = ((target - output)**2).sum()
print(f'Loss: {loss.item()}')

optim = torch.optim.SGD(model.parameters(), lr=0.05)

loss.backward()
optim.step()

for name, param in model.named_parameters():
    print(f'{name} gradient: \n{param.grad}\n\n')
