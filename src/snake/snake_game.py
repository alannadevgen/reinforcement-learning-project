from abc import ABC, abstractmethod
import pygame


class SnakeGame(ABC):
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height

        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
    
    @abstractmethod
    def _place__food(self):
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
    
    @abstractmethod
    def _is_collision(self):
        pass