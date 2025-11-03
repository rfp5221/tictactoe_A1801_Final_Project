import math
import functools
import random
import csv
import time
import os
from concurrent.futures import ProcessPoolExecutor

infinity = math.inf

# -------------------- GLOBAL HELPERS --------------------

def cutoff_depth_fn(max_depth):
    def cutoff(game, state, depth):
        return depth > max_depth
    return cutoff

_state_registry = {}

def hash_state(state):
    h = hash(state)
    _state_registry[h] = state
    return h

def state_from_hash(h):
    return _state_registry.get(h, None)


# -------------------- CORE CLASSES --------------------

class Game:
    def actions(self, state): raise NotImplementedError
    def result(self, state, move): raise NotImplementedError
    def is_terminal(self, state): return not self.actions(state)
    def utility(self, state, player): raise NotImplementedError


class Board(dict):
    empty = '.'
    off = '#'

    def __init__(self, width=5, height=5, to_move=None, **kw):
        dict.__init__(self)
        self.width = width
        self.height = height
        self.to_move = to_move
        self.utility = kw.get('utility', 0)
        self.k = kw.get('k', 4)

    def new(self, changes: dict, **kwds):
        b = Board(width=self.width, height=self.height, to_move=kwds.get('to_move', self.to_move))
        b.update(self)
        b.update(changes)
        b.utility = kwds.get('utility', getattr(self, 'utility', 0))
        b.k = getattr(self, 'k', 4)
        return b

    def __missing__(self, loc):
        x, y = loc
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.empty
        else:
            return self.off

    def __hash__(self):
        return hash((tuple(sorted(self.items())), self.to_move, self.utility))


# -------------------- GAME LOGIC --------------------

def k_in_row(board, player, square, k):
    def count_dir(x, y, dx, dy):
        n = 0
        while (x, y) in board and board[(x, y)] == player:
            n += 1
            x, y = x + dx, y + dy
        return n
    return any(
        count_dir(*square, dx, dy) + count_dir(*square, -dx, -dy) - 1 >= k
        for (dx, dy) in ((0, 1), (1, 0), (1, 1), (1, -1))
    )


class TicTacToe(Game):
    def __init__(self, height=5, width=5, k=4):
        self.k = k
        self.width = width
        self.height = height
        self.squares = {(x, y) for x in range(width) for y in range(height)}
        self.initial = Board(width=width, height=height, to_move='X', utility=0, k=k)

    def actions(self, board):
        return self.squares - set(board.keys())

    def result(self, board, square):
        player = board.to_move
        next_player = 'O' if player == 'X' else 'X'
        new_board = board.new({square: player}, to_move=next_player)
        win = k_in_row(new_board, player, square, self.k)
        new_board.utility = 0 if not win else +1 if player == 'X' else -1
        new_board.k = self.k
        return new_board

    def utility(self, board, player):
        return board.utility if player == 'X' else -board.utility

    def is_terminal(self, board):
        return board.utility != 0 or len(self.squares) == len(board)


# -------------------- SEARCH FUNCTIONS --------------------

def minimax_search_tt_hash(game, state, cutoff, h):
    player = state.to_move

    @functools.lru_cache(maxsize=100000)
    def max_value(state_hash, depth):
        state = state_from_hash(state_hash)
        if state is None or game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(hash_state(game.result(state, a)), depth + 1)
            if v2 > v:
                v, move = v2, a
        return v, move

    @functools.lru_cache(maxsize=100000)
    def min_value(state_hash, depth):
        state = state_from_hash(state_hash)
        if state is None or game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(hash_state(game.result(state, a)), depth + 1)
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(hash_state(state), 0)


def h_alphabeta_search(game, state, cutoff, h):
    player = state.to_move

    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                break
        return v, move

    def min_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta, depth + 1)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                break
        return v, move

    return max_value(state, -infinity, +infinity, 0)


# -------------------- HEURISTICS --------------------

def heuristic_tictactoe(state, player):
    board = state
    opponent = 'X' if player == 'O' else 'O'
    size, k = board.width, board.k

    def line_score(line):
        score = 0
        pc, oc = line.count(player), line.count(opponent)
        if oc == 0 and pc > 0:
            score += 10 ** pc
        if pc == 0 and oc > 0:
            score -= 10 ** oc
        return score

    total = 0
    for r in range(size):
        for c in range(size - k + 1):
            total += line_score([board[(c + i, r)] for i in range(k)])
    for c in range(size):
        for r in range(size - k + 1):
            total += line_score([board[(c, r + i)] for i in range(k)])
    for r in range(size - k + 1):
        for c in range(size - k + 1):
            total += line_score([board[(c + i, r + i)] for i in range(k)])
            total += line_score([board[(c + k - 1 - i, r + i)] for i in range(k)])
    return total


# -------------------- PLAYERS --------------------

def random_player(game, state):
    return random.choice(list(game.actions(state)))

def minimax_hash_player(game, state, cutoff, h):
    return minimax_search_tt_hash(game, state, cutoff, h)[1]

def alphabeta_player(game, state, cutoff, h):
    return h_alphabeta_search(game, state, cutoff, h)[1]

def bayesian_player(game, state):
    player = state.to_move
    opponent = 'O' if player == 'X' else 'O'
    moves = list(game.actions(state))
    width, height, k = state.width, state.height, state.k

    def count_near_k(board, who, move):
        count = 0
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            line = []
            for i in range(-k + 1, k):
                x, y = move[0] + i * dx, move[1] + i * dy
                if 0 <= x < width and 0 <= y < height:
                    line.append(board[(x, y)])
            if line.count(who) == k - 1 and line.count('.'):
                count += 1
        return count

    def feature_probs(board, move):
        x, y = move
        center_dist = math.sqrt((x - width/2)**2 + (y - height/2)**2)
        p_center = math.exp(-0.3 * center_dist)
        near_win = count_near_k(board, player, move)
        p_win = 0.2 + 0.15 * near_win
        block_threat = count_near_k(board, opponent, move)
        p_block = 0.1 + 0.2 * block_threat
        return p_center * p_win * (1 + p_block)

    return max(moves, key=lambda mv: feature_probs(state, mv))


# -------------------- GAME EXECUTION --------------------

def play_game(game, strategies):
    state = game.initial
    while not game.is_terminal(state):
        move = strategies[state.to_move](game, state)
        state = game.result(state, move)
    return state


def run_single_game(args):
    model_name, size, run_id, cutoff_depth = args
    random.seed(run_id)
    game = TicTacToe(height=size, width=size, k=4)
    cutoff = cutoff_depth_fn(cutoff_depth)
    h = heuristic_tictactoe

    if model_name == "minimax_hash":
        model_fn = lambda g, s: minimax_hash_player(g, s, cutoff, h)
    elif model_name == "alphabeta":
        model_fn = lambda g, s: alphabeta_player(g, s, cutoff, h)
    else:
        model_fn = bayesian_player

    strategies = {'X': random_player, 'O': model_fn}
    start = time.time()
    final = play_game(game, strategies)
    runtime = round(time.time() - start, 3)

    # Determine winner label
    if final.utility == 1:
        winner = "random_player"
    elif final.utility == -1:
        winner = model_name
    else:
        winner = "Draw"

    # run_id first in output
    return [run_id, model_name, size*size, runtime, winner]


# -------------------- MAIN --------------------

if __name__ == '__main__':
    os.makedirs("data", exist_ok=True)
    outfile = "data/results.csv"

    with open(outfile, "w", newline="") as f:
        csv.writer(f).writerow(["run_id", "model", "board_size", "runtime_seconds", "winner"])

    models = ["minimax_hash", "alphabeta", "bayesian"]

    for size in range(5, 8): #board sizes highest will go is 7 for quick testing and need to change number to 11 for 10x10
        for model in models:
            print(f"Running {model} on {size}x{size}...")
            args_list = [(model, size, i, 3) for i in range(1, 3)] #number of games to run per model reduced to 2 for testing but change to 11

            with ProcessPoolExecutor() as ex:
                results = list(ex.map(run_single_game, args_list))

            with open(outfile, "a", newline="") as f:
                csv.writer(f).writerows(results)

            avg_time = sum(r[3] for r in results) / len(results)
            print(f"{model} done | Avg time: {avg_time:.2f}s\n")