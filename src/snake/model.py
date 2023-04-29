import torch
import torch.nn as nn
import torch.nn.functional as functional
import os


class Model(nn.Module):
    '''
    # Pour l'instant, je ne vois pas vraiment comment décrire cette classe.

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
        # Pas de description pour l'instant
    save(file_name)
        Save the model in a folder
    
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.first_layer = nn.Linear(input_size, hidden_size).cuda()
        self.second_layer = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, step):
        '''
        # Idem, je pense que je n'ai pas tout compris à cette fonction.

        Parameters
        ----------
        step 

        Returns
        -------
        step : nn.Linear

        '''
        step = functional.relu(self.first_layer(step))
        step = self.second_layer(step)
        return step

    def save(self, file_name='model.pth'):
        '''
        Save the model in a folder.

        Parameters
        ----------
        file_name : path, optional
            name to give to the file saved
        '''
        folder_path = 'src/model'
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)
