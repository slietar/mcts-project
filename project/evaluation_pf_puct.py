import math
import random
from copy import deepcopy
import numpy as np
from game import Backgammon
from mcts_puct import PUCTMCTS
from mcts_pf import IPFTMCTS, basic_monte_carlo


def evaluate_ipft_vs_monte_carlo(num_games=10, num_simulations=100):
    ipft_wins = 0
    mc_wins = 0

    for game_index in range(num_games):
        print(f"Starting Game {game_index + 1}...")
        game = Backgammon()
        mcts_ipft = IPFTMCTS(simulations=num_simulations, num_particles=20, lambda_value=0.1)
        mcts_puct = PUCTMCTS(c_puct=1.0, simulations=num_simulations)

        while not (game.p0_won() or game.p1_won()):
            if game.turn_p0:  # IPFT turn
                move = mcts_ipft.run(game)
            else:  # Basic Monte Carlo turn
                move = basic_monte_carlo(game, num_simulations)

            if move is None:
                break

            game.play(*move)

        if game.p0_won():
            ipft_wins += 1
        elif game.p1_won():
            mc_wins += 1

        print(f"Game {game_index + 1} Completed. Winner: {'IPFT' if game.p0_won() else 'Monte Carlo'}")

    ipft_win_rate = (ipft_wins / num_games) * 100
    mc_win_rate = (mc_wins / num_games) * 100

    print(f"\nResults after {num_games} games:")
    print(f"IPFT Win Rate: {ipft_win_rate}%")
    print(f"Monte Carlo Win Rate: {mc_win_rate}%")


if __name__ == "__main__":
    evaluate_ipft_vs_monte_carlo(num_games=10, num_simulations=100)
