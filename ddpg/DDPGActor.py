import torch
import torch.nn as nn
import torch.nn.functional as func


class DDPGActor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DDPGActor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = func.relu(self.linear1(s))
        x = func.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x



