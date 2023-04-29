import click
import os
from src.snake.agent import Agent
from src.snake.snake_game_ai import SnakeGameAI
from src.utils.plot import plot

os.environ["SDL_VIDEODRIVER"] = "dummy"


@click.command()
@click.option(
    '--type',
    default='AI',
    help='Sample size.',
    type=click.Choice(['AI', 'HUMAN'], case_sensitive=False)
)
def main(type):
    if type == "AI":
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        record = 0
        agent = Agent()
        game = SnakeGameAI()
        
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
                agent.nb_game += 1
                agent.train_long_memory()
                if score > reward:  # new High score
                    reward = score
                    agent.model.save()
                print(f'Game #{agent.nb_game} Score: {score} Record: {record}')
                
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.nb_game
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    main()
