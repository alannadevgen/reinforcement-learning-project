import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


class Model(nn.Module):
    '''
    Creates a personalised neural net model for the snake.

    Attributes
    ---------
    input_size : int
        size of the input
    hidden_size : int
        size of the input in the second layer
    output_size : int
        size of the output

    Methods
    -------
    forward(step)
        Apply a forward step in the neural net.
    save(file_name)
        Save the model in a folder
    
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        '''
        Applies a linear transformation to the incoming data.
        
        input_size: int
            Size of the input layer
        hidden_size: int
            Size of the hidden layer
        output_size: int
            Size of the output layer
        '''
        super().__init__()
        self.first_layer = nn.Linear(input_size, hidden_size).cuda()
        self.second_layer = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, step):
        '''
        Applies the rectified linear unit function element-wise
        then a linear step.

        Parameters
        ----------
        step: layer
            Layer to apply

        Returns
        -------
        step : nn.Linear
            Tensor

        '''
        step = functional.relu(self.first_layer(step))
        step = self.second_layer(step)
        return step

    def save(self, folder_path = 'src/model', file_name='model.pth'):
        '''
        Save the model in a file.

        Parameters
        ----------
        folder_path : path, optional
            name of the folder
        file_name : path, optional
            name to give to the file saved
        '''
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)
