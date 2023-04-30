from abc import ABC, abstractmethod
import pygame


class SnakeGame(ABC):
    '''
    Defines the structure of a snake game.

    Attributes
    ----------
    width : int
        width of screen
    height : int
        height of screen
    display : pygame.display
        display of the screen game
    clock : pygame.time.Clock()
        define the speed of the game


    Methods
    -------
    place_food()
        Place food on the screen
    play_step()
        Collect imput to make the snake move
    update_ui()
        Update the game
    move()
        Define how the snake move
    '''
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()

    @abstractmethod
    def _place_food(self):
        pass

    @abstractmethod
    def play_step(self):
        pass

    @abstractmethod
    def _update_ui(self):
        pass

    @abstractmethod
    def _move(self):
        pass
