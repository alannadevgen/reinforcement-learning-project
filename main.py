import click
import os
import pygame
from src.snake.agent import Agent
from src.snake.snake_game_ai import SnakeGameAI
from src.snake.snake_game_human import SnakeGameHuman
from src.utils.plot import plot

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@click.command()
@click.option(
    '--type',
    default='AI',
    help='Type of game to compute.',
    type=click.Choice(['AI', 'HUMAN'], case_sensitive=False)
)
@click.option(
    '--speed',
    default=20,
    help='Snake speed',
    type=int
)
def main(type, speed):
    if type.upper() == "AI":
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        highscore = 0
        agent = Agent()
        game = SnakeGameAI(speed=speed)

        while True:
            # Get Old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(
                state_old, final_move, reward, state_new, done
            )

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # Train long memory,plot result
                game.reset()
                agent.num_game += 1
                agent.train_long_memory()
                if score > highscore:  # new High score
                    highscore = score
                    agent.model.save()
                print(
                    f'Game #{agent.num_game}\t'
                    f'\tScore: {score}'
                    f'\tHigh score: {highscore}'
                    )

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.num_game
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores, highscore=highscore)

    elif type.upper() == "HUMAN":
        game = SnakeGameHuman(speed=speed)

        # Game loop
        # game_over=False
        while True:
            game_over, score = game.play_step()
            if game_over:
                break
        print('Final Score', score)

        pygame.quit()


if __name__ == "__main__":
    main()
