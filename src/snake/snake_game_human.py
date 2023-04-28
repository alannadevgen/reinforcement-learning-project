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
    def __init__(self, width=640, height=480):
        super().__init__()

        # init game state
        self.direction = DIRECTION.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place__food()

    def _place__food(self):
        x = random.randint(0, (self.width-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place__food()

    def play_step(self):
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
        if self.head == self.food:
            self.score += 1
            self._place__food()
        else:
            self.snake.pop()
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. Return game Over and Display Score
        return game_over, self.score

    def _update_ui(self):
        self.display.fill(COLORS.BLACK.value)
        for pt in self.snake:
            pygame.draw.rect
            (
                self.display, COLORS.BLUE, pygame.Rect
                (
                    pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE
                )
            )
            pygame.draw.rect(
                self.display, COLORS.PURPLE, pygame.Rect(pt.x+4, pt.y+4, 12, 12)
            )
        pygame.draw.rect(
            self.display, COLORS.RED, pygame.Rect(
                self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE
                )
        )
        text = font.render(
            "Score: " + str(self.score), True, COLORS.WHITE.value
        )
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
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
        # hit boundary
        if (self.head.x > self.width-BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0):
            return True
        if (self.head in self.snake[1:]):
            return True
        return False


if __name__ == "__main__":
    game = SnakeGameHuman()

    # Game loop
    # game_over=False
    while True:
        game_over, score = game.play_step()
        if game_over:
            break
    print('Final Score', score)

    pygame.quit()
