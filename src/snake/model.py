import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.first_layer = nn.Linear(input_size, hidden_size).cuda()
        self.second_layer = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, step):
        step = functional.relu(self.first_layer(step))
        step = self.second_layer(step)
        return step

    def save(self, file_name='model.pth'):
        folder_path = 'src/model'
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)
