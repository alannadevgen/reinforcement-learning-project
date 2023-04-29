from collections import deque
import numpy as np
import random
import torch
from src.snake.snake_game_ai import Point
from src.utils.utils import MAX_MEMORY, BATCH_SIZE, LR, BLOCK_SIZE
from src.enum.direction import DIRECTION
from src.snake.model import Model
from src.snake.trainer import Trainer


class Agent:
    '''
    Define all the components of an agent playing the game.

    Attributes
    ---------
    nb_game : int
        number of games performed by the agent
    epsilon : int
        parameter for randomness
    gamma  : int
        parameter for discount rate
    memory : deque
        abstract data type that generalizes a queue, where elements can be removed if they are at the front or back
    model : Model
        model of the game
    trainer : Trainer
        trainer of the AI

    Methods
    -------
    get_state(game)
        Get the state of the game
    remember(state, action, reward, next_state, done)
        Add elements to the AI memory
    train_long_memory()
        Train the long term memory of the AI
    train_short_memory(state, action, reward, next_state, done)
        Train the short term memory of the AI
    get_action(state)
        Get an action to be performed according to the state
    '''
    def __init__(self):
        self.nb_game = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Model(11, 256, 3)
        self.trainer = Trainer(self.model, lr=LR, gamma=self.gamma)

    # state (11 Values)
    # [ danger straight, danger right, danger left,
    # direction left, direction right,
    # direction up, direction down
    # food left,food right,
    # food up, food down]
    @staticmethod
    def get_state(game):
        '''
        Get the current state of the game

        Parameters
        ----------
        game : SnakeGame
            object representing the game

        Returns
        -------
        state : np.array
            state of the game
        '''
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        dir_left = game.direction == DIRECTION.LEFT
        dir_right = game.direction == DIRECTION.RIGHT
        dir_up = game.direction == DIRECTION.UP
        dir_down = game.direction == DIRECTION.DOWN

        state = [
            # Danger Straight
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_right)),

            # Danger right
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            # Danger Left
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)),

            # Move Direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # Food Location
            game.food.x < game.head.x,  # food is in left
            game.food.x > game.head.x,  # food is in right
            game.food.y < game.head.y,  # food is up
            game.food.y > game.head.y   # food is down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        '''
        Append to the memory of the AI the change of state, the action and the reward.

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
        done : ?
            # Je ne vois pas à quoi ce paramètre sert
        '''
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if memory exceed

    def train_long_memory(self):
        '''
        Train the long term memory of the AI.
        '''
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        '''
        Train the short term memory of the AI.

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
        done : ?
            # Je ne vois pas à quoi ce paramètre sert
        '''
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        '''
        Define the action which is going to serve as a future input.

        Parameters
        ---------
        state : np.array
            state of the game

        Returns
        -------
        final_move : list
            move to be performed
        '''
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.nb_game
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).cuda()
            prediction = self.model(state0).cuda()  # prediction by model
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
