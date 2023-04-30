import pygame
import random

from src.enum.direction import DIRECTION
from src.enum.colors import COLORS
from src.utils.utils import BLOCK_SIZE, SPEED, Point
from src.snake.snake_game import SnakeGame

pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Reset
# Reward
# Play(action) -> Direction
# Game_Iteration
# is_collision


class SnakeGameHuman(SnakeGame):
    '''
    This class makes the snake game playable for human users.

    Parameters
    ----------
    SnakeGame : SnakeGame
        structure of the game

    Attributes
    ----------
    width : int
        width of screen
    height : int
        height of screen
    direction : DIRECTION
        current direction of the snake
    head : Point
        head of the snake
    snake : list
        full body of snake
    score : int
        score of the game
    food1 : Point
        position of the food number 1
    food2 : Point
        position of the food number 2
    type_food1 : int
        type of the food number 1 
    type_food2 : int
        type of the food number 2 

    Methods
    -------
    place_food()
        Define type of foods and place them on the screen
    play_step()
        Collect user imput to make the snake move
    update_ui()
        Update the game
    move()
        Define how the snake move
    is_collision()
        Define if the snake crashed
    '''
    def __init__(self, width=640, height=480, SPEED=SPEED):
        '''
        Parameters
        ----------
        width : int, optional
            width of screen (default is 640)
        height : int, optional
            height of screen (default is 480)
        '''
        super().__init__(width=width, height=height)

        # init game state
        self.SPEED = SPEED
        self.direction = DIRECTION.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food1 = None
        self.food2 = None
        self.type_food1 = None
        self.type_food2 = None
        self._place__food()
        

    def _place__food(self):
        '''
        Place foods at the beginning of the game or after it has been eaten.
        '''
        self.type_food1 = random.randint(1,3)
        self.type_food2 = random.randint(1,3)
        x1 = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y1 = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food1 = Point(x1, y1)
        x2 = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y2 = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food2 = Point(x2, y2)
        if (self.food1 in self.snake) | (self.food2 in self.snake):
            self._place__food()
        

    def play_step(self):
        '''
        Collect the user input, in order to define the movement of the snake.
        Checks if there is a game over.
        '''
        # 1. Collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = DIRECTION.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = DIRECTION.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = DIRECTION.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = DIRECTION.DOWN
        # 2. Move
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # 3. Check if game Over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        # 4. Place new Food or just move
        if (self.head == self.food1) :  
            if self.type_food1 == 1 :
                self.score += 10
            elif self.type_food1 == 2 :
                self.score += 5
            else :
                self.score += 1
            self._place__food()
        elif (self.head == self.food2) :  
            if self.type_food2 == 1 :
                self.score += 10
            elif self.type_food2 == 2 :
                self.score += 5
            else :
                self.score += 1
            self._place__food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(self.SPEED)
        # 6. Return game Over and Display Score
        return game_over, self.score

    def _update_ui(self):
        '''
        Update the game after each action.
        '''
        self.display.fill(COLORS.BLACK.value)
        for pt in self.snake:
            pygame.draw.rect
            (
                self.display, COLORS.BLUE.value, pygame.Rect
                (
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE
                )
            )
            pygame.draw.rect(
                self.display, COLORS.PURPLE.value, pygame.Rect(pt.x+4, pt.y+4, 12, 12)
            )

        if self.type_food1 == 1 :
                pygame.draw.rect(
                    self.display, COLORS.GREEN.value, pygame.Rect(
                    self.food1.x, self.food1.y, BLOCK_SIZE, BLOCK_SIZE)
                )
        elif self.type_food1 == 2 :
                pygame.draw.rect(
                    self.display, COLORS.YELLOW.value, pygame.Rect(
                    self.food1.x, self.food1.y, BLOCK_SIZE, BLOCK_SIZE)
                )
        else :
                pygame.draw.rect(
                    self.display, COLORS.RED.value, pygame.Rect(
                    self.food1.x, self.food1.y, BLOCK_SIZE, BLOCK_SIZE)
                    )
        if self.type_food2 == 1 :
                pygame.draw.rect(
                    self.display, COLORS.GREEN.value, pygame.Rect(
                    self.food2.x, self.food2.y, BLOCK_SIZE, BLOCK_SIZE)
                )
        elif self.type_food2 == 2 :
                pygame.draw.rect(
                    self.display, COLORS.YELLOW.value, pygame.Rect(
                    self.food2.x, self.food2.y, BLOCK_SIZE, BLOCK_SIZE)
                )
        else :
                pygame.draw.rect(
                    self.display, COLORS.RED.value, pygame.Rect(
                    self.food2.x, self.food2.y, BLOCK_SIZE, BLOCK_SIZE)
                    )

        text = font.render(
            "Score: " + str(self.score), True, COLORS.WHITE.value
        )
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        '''
        Define how the snake is going to move.
        '''
        x = self.head.x
        y = self.head.y
        if direction == DIRECTION.RIGHT:
            x += BLOCK_SIZE
        elif direction == DIRECTION.LEFT:
            x -= BLOCK_SIZE
        elif direction == DIRECTION.DOWN:
            y += BLOCK_SIZE
        elif direction == DIRECTION.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def _is_collision(self):
        '''
        Define if there is a collision between the snake and the boundaries or its body.
        '''
        # hit boundary
        if (self.head.x > self.width-BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0):
            return True
        if (self.head in self.snake[1:]):
            return True
        return False
