import torch
import torch.nn as nn
import torch.optim as optim
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""


class Trainer:
    '''
    Trains the AI to make a decision at each step of the game according to previous results.

    Attributes
    ----------
    lr : float
        learning rate
    gamma : int
        parameter for discount rate
    model : Model
        Neural net model
    optimer : optim.Adam
        optimize parameters of the model
    criterion : loss function, default is nn.MSELoss()
        loss function used as a criterion for efficiency

    Methods
    -------
    train_step(state, action, reward, next_state, done)
        Train the model at each step

    '''
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        # for i in self.model.parameters():
        #     print(i.is_cuda)

    def train_step(self, state, action, reward, next_state, done):
        '''
        Train the model after each action according to the results.

        Parameters
        ---------
        state : np.array
            state of the game before the action
        action : list
            action to be taken by the AI
        reward : int
            reward of the action
        next_state : np.array
            state of the game after the action
        done : bool
            Is the game over?
        '''
        state = torch.tensor(state, dtype=torch.float).cpu()
        next_state = torch.tensor(next_state, dtype=torch.float).cpu()
        action = torch.tensor(action, dtype=torch.long).cpu()
        reward = torch.tensor(reward, dtype=torch.float).cpu()

        if len(state.shape) == 1:  # only one parameter to train ,
            # Hence convert to tuple of shape (1, x)
            state = torch.unsqueeze(state, 0).cpu()
            next_state = torch.unsqueeze(next_state, 0).cpu()
            action = torch.unsqueeze(action, 0).cpu()
            reward = torch.unsqueeze(reward, 0).cpu()
            done = (done, )

        # 1. Predicted Q value with current state
        pred = self.model(state).cpu()
        target = pred.clone().cpu()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])).cpu()
            target[idx][torch.argmax(action).item()] = Q_new
        
        # 2. Q_new = reward + gamma * max(next_predicted Qvalue)
        # -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimer.step()
