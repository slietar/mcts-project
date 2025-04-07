import math
import random
from copy import deepcopy
import numpy as np
from game import Backgammon  # Ensure you import the Backgammon class from your game.py


def particle_filter_update(particles, action, observation, num_particles=20):
    """
    Updates the particle set given an action and an observation using a Gaussian kernel.
    """
    updated_particles = []

    for particle in particles:
        new_particle = deepcopy(particle)
        start_column, distance = action

        # Check if the action is legal before applying
        legal_moves = new_particle.legal_moves(distance)
        if start_column < len(legal_moves) and legal_moves[start_column]:
            try:
                new_particle.play(start_column, distance)
                if new_particle.check_integrity():
                    updated_particles.append(new_particle)
            except AssertionError:
                continue  # Skip particles that cause an integrity error

    # If no valid particles were generated, return the original particles
    if len(updated_particles) == 0:
        return particles

    # Resampling
    weights = np.array([np.exp(-np.linalg.norm(p.board - observation)) for p in updated_particles])
    if np.sum(weights) == 0:
        return particles  # If all weights are zero, return original particles

    weights /= np.sum(weights)
    indices = np.random.choice(range(len(updated_particles)), size=num_particles, p=weights)
    resampled_particles = [updated_particles[i] for i in indices]

    return resampled_particles


def estimate_entropy(particles):
    """
    Estimates entropy using Kernel Density Estimation (KDE).
    """
    positions = np.array([p.board for p in particles])
    entropy = -np.mean([np.log(np.mean(np.exp(-np.linalg.norm(positions - pos, axis=1)))) for pos in positions])

    return entropy


class Node:
    def __init__(self, particles, parent=None):
        self.particles = particles
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False
        self.entropy = estimate_entropy(particles)

    def expand(self, legal_moves):
        for move in legal_moves:
            if move not in self.children:
                self.children[move] = Node(particles=[], parent=self)
        self.is_expanded = True

    def is_leaf(self):
        return not self.is_expanded

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class IPFTMCTS:
    def __init__(self, simulations=800, num_particles=20, lambda_value=0.1):
        self.simulations = simulations
        self.num_particles = num_particles
        self.lambda_value = lambda_value

    def run(self, root_game):
        particles = [deepcopy(root_game) for _ in range(self.num_particles)]
        root_node = Node(particles)

        for _ in range(self.simulations):
            node = root_node
            particles = deepcopy(root_node.particles)

            while not node.is_leaf():
                move = random.choice(list(node.children.keys()))
                observation = np.array([p.board for p in particles]).mean(axis=0)
                particles = particle_filter_update(particles, move, observation=observation, num_particles=self.num_particles)
                node = node.children[move]

            if not root_game.p0_won() and not root_game.p1_won():
                legal_moves = self.get_legal_moves(particles[0])
                node.expand(legal_moves)

            value = self.simulate(particles)

            self.backpropagate(node, value)

        return self.best_action(root_node)

    def simulate(self, particles):
        value = random.choice([-1, 1])
        return value

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective for opponent
            node = node.parent

    def get_legal_moves(self, game):
        legal_moves = {}
        for distance in range(1, 7):
            moves = game.legal_moves(distance).nonzero()[0]
            for move in moves:
                legal_moves[(move, distance)] = 1 / len(moves) if len(moves) > 0 else 0
        return legal_moves

    def best_action(self, root_node):
        best_visit_count = -1
        best_move = None

        for move, child in root_node.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_move = move

        return best_move
    
def basic_monte_carlo(game: Backgammon, num_simulations: int) -> int:
    """
    Perform basic Monte Carlo simulation from the given game state.
    Returns the most promising move based on win rate.
    """
    dice_values = [1, 2, 3, 4, 5, 6]
    move_scores = {}

    for dice_value in dice_values:
        legal_moves = game.legal_moves(dice_value).nonzero()[0]

        if len(legal_moves) == 0:
            continue

        for move in legal_moves:
            wins = 0

            for _ in range(num_simulations):
                simulated_game = deepcopy(game)
                simulated_game.play(move, dice_value)

                while not (simulated_game.p0_won() or simulated_game.p1_won()):
                    current_player = 0 if simulated_game.turn_p0 else 1
                    dice_roll = random.randint(1, 6)
                    legal_moves = simulated_game.legal_moves(dice_roll).nonzero()[0]

                    if len(legal_moves) > 0:
                        selected_move = random.choice(legal_moves)
                        simulated_game.play(selected_move, dice_roll)
                    else:
                        break

                if simulated_game.p0_won():
                    wins += 1

            win_rate = wins / num_simulations
            move_scores[(move, dice_value)] = win_rate

    # Return the move with the highest win rate
    if move_scores:
        best_move = max(move_scores, key=move_scores.get)
        return best_move
    else:
        return 


def evaluate_ipft_vs_monte_carlo(num_games=10, num_simulations=100):
    ipft_wins = 0
    mc_wins = 0

    for game_index in range(num_games):
        print(f"Starting Game {game_index + 1}...")
        game = Backgammon()
        mcts = IPFTMCTS(simulations=num_simulations, num_particles=20, lambda_value=0.1)

        while not (game.p0_won() or game.p1_won()):
            if game.turn_p0:  # IPFT turn
                move = mcts.run(game)
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
    game = Backgammon()
    mcts = IPFTMCTS(simulations=100, num_particles=20, lambda_value=0.1)

    while not (game.p0_won() or game.p1_won()):
        move = mcts.run(game)
        if move is None:
            break
        game.play(*move)
        game.print()

    print("Winner:", game.winner())
    
    # Evaluate the performance of IPFT vs Basic Monte Carlo
    
    evaluate_ipft_vs_monte_carlo(num_games=10, num_simulations=100)
    
