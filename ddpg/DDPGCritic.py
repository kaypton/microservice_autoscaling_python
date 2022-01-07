import torch
import torch.nn as nn
import torch.nn.functional as func


class DDPGCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DDPGCritic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = func.relu(self.linear1(x))
        x = func.relu(self.linear2(x))
        x = self.linear3(x)

        return x


