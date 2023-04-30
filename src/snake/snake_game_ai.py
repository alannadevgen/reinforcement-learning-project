import pygame
import random
import numpy as np

from src.enum.direction import DIRECTION
from src.enum.colors import COLORS
from src.utils.utils import BLOCK_SIZE, SPEED, Point
from src.snake.snake_game import SnakeGame

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class SnakeGameAI(SnakeGame):
    '''
    This class plays snake using an AI.

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
    food : Point
        position of the food
    type_food : int
        type of the food
    frame_iteration : int
        number of iterations

    Methods
    -------
    reset()
        Reset the game
    place_food()
        Define type of rhe food and place it on the screen
    play_step()
        Collect AI imput to make the snake move
    update_ui()
        Update the game frame
    move()
        Define how the snake moves
    is_collision()
        Define if the snake crashed against a wall
    '''
    def __init__(self, width=640, height=480, speed=SPEED):
        '''
        Parameters
        ----------
        width : int, optional
            width of screen (default is 640)
        height : int, optional
            height of screen (default is 480)
        '''
        super().__init__(width, height)
        self.speed = speed

        # init game state
        self.reset()

    def reset(self):
        '''
        Reset the game back to its original state.
        '''
        self.direction = DIRECTION.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self.type_food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        '''
        Place food at the beginning of the game and after it has been eaten.
        '''
        x = random.randint(
            0, (self.width - BLOCK_SIZE)//BLOCK_SIZE
        ) * BLOCK_SIZE
        y = random.randint(
            0, (self.height - BLOCK_SIZE)//BLOCK_SIZE
        ) * BLOCK_SIZE
        self.food = Point(x, y)
        if (self.food in self.snake):
            self._place_food()

    def play_step(self, action):
        '''
        Collect the AI input, in order to define the movement of the snake.
        The input is defined by a reward.
        Checks if there is a game over.

        Parameters
        ----------
        action : list
            action taken by the AI

        Returns
        -------
        reward : int
            Reward of the step/action
        game_over : bool
            Define if the game is over
        score : int
            Score of the game after the step
        '''
        self.frame_iteration += 1
        # 1. Collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game Over
        reward = 0  # eat food: +10 , game over: -10 , else: 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        # 4. Place new Food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()

        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(self.speed)

        # 6. Return game Over and display score
        return reward, game_over, self.score

    def _update_ui(self):
        '''
        Update the game after each action.
        '''
        self.display.fill(COLORS.BLACK.value)
        for pt in self.snake:
            pygame.draw.rect(
                self.display,
                COLORS.BLUE.value,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect(
                self.display,
                COLORS.PURPLE.value,
                pygame.Rect(pt.x+4, pt.y+4, 12, 12)
            )
            pygame.draw.rect(
                self.display, COLORS.RED.value, pygame.Rect(
                    self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
                )
        # if self.type_food == 1:
        #     pygame.draw.rect(
        #         self.display, COLORS.GREEN.value, pygame.Rect(
        #             self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        #     )
        # elif self.type_food == 2:
        #     pygame.draw.rect(
        #         self.display, COLORS.YELLOW.value, pygame.Rect(
        #             self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        #     )
        # else:
        #     pygame.draw.rect(
        #         self.display, COLORS.RED.value, pygame.Rect(
        #             self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        #         )
        text = font.render(
            "Score: " + str(self.score), True, COLORS.WHITE.value
        )
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        '''
        Define how the snake is going to move.

        Parameters
        ----------
        action : list
            action taken by the AI
        '''
        # Action
        # [1,0,0] -> Straight
        # [0,1,0] -> Right Turn
        # [0,0,1] -> Left Turn

        clock_wise = [
            DIRECTION.RIGHT, DIRECTION.DOWN, DIRECTION.LEFT, DIRECTION.UP
        ]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right Turn
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left Turn
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == DIRECTION.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == DIRECTION.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == DIRECTION.DOWN:
            y += BLOCK_SIZE
        elif self.direction == DIRECTION.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)

    def is_collision(self, pt=None):
        '''
        Define if there is a collision between the snake and the boundaries
        or its body.
        '''
        if pt is None:
            pt = self.head
        # hit boundary
        if pt.x > self.width-BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
