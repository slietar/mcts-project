import math
import random
from copy import deepcopy
import numpy as np
from game import Backgammon  # Ensure you import the Backgammon class from your game.py
    


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
            return None



class Node:
    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.is_expanded = False

    def expand(self, legal_moves):
        for move, prior in legal_moves.items():
            if move not in self.children:
                self.children[move] = Node(state=None, parent=self, prior=prior)
        self.is_expanded = True

    def is_leaf(self):
        return not self.is_expanded

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            ucb = child.value() + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            if ucb > best_score:
                best_score = ucb
                best_move = move
                best_child = child

        return best_move, best_child


class PUCTMCTS:
    def __init__(self, c_puct=1.0, simulations=800):
        self.c_puct = c_puct
        self.simulations = simulations

    def run(self, root_state):
        root_node = Node(state=root_state)

        for _ in range(self.simulations):
            node = root_node
            state = root_state.copy()

            # Selection
            while not node.is_leaf():
                move, node = node.select_child(self.c_puct)
                state.play(*move)

            # Expansion
            if not state.p0_won() and not state.p1_won():
                legal_moves = self.get_legal_moves_with_priors(state)
                node.expand(legal_moves)

            # Simulation
            value = self.simulate(state)

            # Backpropagation
            self.backpropagate(node, value)

        return self.best_action(root_node)

    def simulate(self, state):
        while not state.p0_won() and not state.p1_won():
            legal_moves = self.get_legal_moves_with_priors(state)
            
            if not legal_moves:  # Check if there are no legal moves
                return 0  # Return a draw if the game is stuck or over
            
            move = random.choice(list(legal_moves.keys()))  # Use random.choice() instead of np.random.choice()
            state.play(*move)
            
        return 1 if state.p0_won() else -1 if state.p1_won() else 0

    def backpropagate(self, node, value):
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            value = -value  # Switch perspective for opponent
            node = node.parent

    def get_legal_moves_with_priors(self, state):
        legal_moves = {}
        for distance in range(1, 7):
            moves = state.legal_moves(distance).nonzero()[0]
            for move in moves:
                legal_moves[(move, distance)] = 1 / len(moves)  # Uniform priors for now
        return legal_moves

    def best_action(self, root_node):
        best_visit_count = -1
        best_move = None

        for move, child in root_node.children.items():
            if child.visit_count > best_visit_count:
                best_visit_count = child.visit_count
                best_move = move

        return best_move
    
def evaluate_mcts_vs_random(num_games: int, iterations: int, c_puct: float):
    wins = {0: 0, 1: 0}

    for game_index in range(num_games):
        game = Backgammon()
        mcts_agent = PUCTMCTS(c_puct=c_puct, simulations=iterations)

        while not (game.p0_won() or game.p1_won()):
            if game.turn_p0:
                # Player 0 (MCTS PUCT) Turn
                move = mcts_agent.run(game)  # Run MCTS directly on the main `game`
                
                if move is None:
                    break  # No valid move, terminate game

                game.play(*move)
            else:
                # Player 1 (Random Play) Turn
                random_move = None
                for dice_roll in range(1, 7):  # Try all possible dice rolls (1 to 6)
                    legal_moves = game.legal_moves(dice_roll).nonzero()[0]
                    if len(legal_moves) > 0:
                        move = random.choice(legal_moves)
                        random_move = (move, dice_roll)
                        break  # Take the first legal move we find
                
                if random_move is None:
                    break  # No valid move, terminate game
                
                game.play(*random_move)

        winner = game.winner()
        if winner is not None:
            wins[winner] += 1

        print(f"Game {game_index + 1} completed. Winner: {'Player 0 (MCTS PUCT)' if winner == 0 else 'Player 1 (Random Player)'}")

    print(f"\nEvaluation Results over {num_games} games:")
    print(f"Player 0 (MCTS PUCT) Wins: {wins[0]}")
    print(f"Player 1 (Random Player) Wins: {wins[1]}")



if __name__ == "__main__":
    game = Backgammon()
    mcts = PUCTMCTS(c_puct=1.0, simulations=100)

    while not (game.p0_won() or game.p1_won()):
        move = mcts.run(game)
        if move is None:
            break
        game.play(*move)
        game.print()

    print("Winner:", game.winner())

    # Evaluate the performance of Basic Monte Carlo vs MCTS
    evaluate_mcts_vs_random(num_games=10, iterations=100, c_puct=1.0)    