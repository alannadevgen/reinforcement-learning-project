import pygame
import random
import numpy as np

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



class SnakeGameAI(SnakeGame):
    def __init__(self, width=640, height=480):
        super().__init__()

        # init game state
        self.reset()

    def reset(self):
        self.direction = DIRECTION.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()
        self.frame_iteration = 0

    def _place__food(self):
        x = random.randint(
            0, (self.width - BLOCK_SIZE)//BLOCK_SIZE
        ) * BLOCK_SIZE
        y = random.randint(
            0, (self.height - BLOCK_SIZE)//BLOCK_SIZE
        ) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place__food()

    def play_step(self, action):
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
            self._place__food()

        else:
            self.snake.pop()

        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score

        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill(COLORS.BLACK.value)
        for pt in self.snake:
            pygame.draw.rect
            (
                self.display,
                COLORS.BLUE.value,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )
            pygame.draw.rect
            (
                self.display,
                COLORS.PURPLE.value,
                pygame.Rect(pt.x+4, pt.y+4, 12, 12)
            )
        pygame.draw.rect
        (
            self.display,
            COLORS.RED.value,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        )
        text = font.render(
            "Score: " + str(self.score), True, COLORS.WHITE.value
        )
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
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
        if pt is None:
            pt = self.head
        # hit boundary
        if pt.x > self.width-BLOCK_SIZE or pt.x < 0 or pt.y > self.height - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False
