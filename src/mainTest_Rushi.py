import math
import functools
import random
from collections import defaultdict

infinity = math.inf

# Use lru_cache for pure functions (used in minimax example).
cache = functools.lru_cache(maxsize=10**6)

def minimax_player(game, state):
    return minimax_search_tt(game, state)

def alphabeta_player(game, state, cutoff, h):
    return h_alphabeta_search(game, state, cutoff=cutoff, h=h)

def random_player(game, state):
    """Randomly select a legal move."""
    return random.choice(list(game.actions(state)))


def player(strategy_fn):
    """Return a callable player that fits play_game() interface."""
    return lambda g, s: strategy_fn(g, s)

def minimax_search_tt(game, state):
    """Plain minimax (value, move). Caches on state because functions take only state."""
    player = state.to_move

    @functools.lru_cache(maxsize=100000)
    def max_value(state_hash):
        state = state_from_hash(state_hash)
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(hash_state(game.result(state, a)))
            if v2 > v:
                v, move = v2, a
        return v, move

    @functools.lru_cache(maxsize=100000)
    def min_value(state_hash):
        state = state_from_hash(state_hash)
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(hash_state(game.result(state, a)))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(hash_state(state))


def cutoff_depth(d):
    """A cutoff function that searches to depth d (stop when depth > d)."""
    return lambda game, state, depth: depth > d


def h_alphabeta_search(game, state, cutoff=cutoff_depth(6), h=lambda s, p: 0):
    """
    Alpha-beta with heuristic cutoff.
    This implementation DOES NOT use the previous unsafe cache1; it does not cache alpha/beta
    results (unsafe to cache without including alpha/beta/depth in key).
    Returns (value, move).
    """
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
                return v, move
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
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity, 0)


def heuristic_5x5_tictactoe(state, player):
    """
    Heuristic evaluation for a (possibly 5x5) board.
    Uses the board.empty sentinel and counts player/opponent pieces in each k-length window.
    More weight for longer contiguous presence in a line.
    """
    board = state  # state is Board
    opponent = 'X' if player == 'O' else 'O'
    size = board.width  # assume square or use width/height
    k = board.k if hasattr(board, "k") else size  # fallback; TicTacToe sets k in game, but not board

    def line_score(line):
        # line elements are values like 'X', 'O', or board.empty ('.')
        score = 0
        # count player's pieces and empties
        pc = line.count(player)
        oc = line.count(opponent)
        ec = line.count(board.empty)
        # if line contains only player and empties -> positive
        if oc == 0 and pc > 0:
            score += (10 ** pc)
        # if line contains only opponent and empties -> negative
        if pc == 0 and oc > 0:
            score -= (10 ** oc)
        return score

    total_score = 0
    # Evaluate all k-length windows in rows, columns, and diagonals (sliding windows)
    # This is more complete than only the two main diagonals.
    # Rows:
    for r in range(size):
        for c in range(size - k + 1):
            line = [board[(c + i, r)] for i in range(k)]
            total_score += line_score(line)
    # Columns:
    for c in range(size):
        for r in range(size - k + 1):
            line = [board[(c, r + i)] for i in range(k)]
            total_score += line_score(line)
    # Diagonals (down-right):
    for r in range(size - k + 1):
        for c in range(size - k + 1):
            line = [board[(c + i, r + i)] for i in range(k)]
            total_score += line_score(line)
    # Anti-diagonals (down-left):
    for r in range(size - k + 1):
        for c in range(k - 1, size):
            line = [board[(c - i, r + i)] for i in range(k)]
            total_score += line_score(line)

    return total_score


# ---------- Helpers for caching minimax using a hash transformation ----------
# We'll provide simple hash -> state and state -> hash helpers using Board.__hash__ and a dict.
# This is a convenience so that the lru_cache for minimax (which requires hashable args)
# can operate with integers instead of Board objects.

_state_registry = {}

def hash_state(state):
    """Return a stable integer key for the state and register the state."""
    h = hash(state)
    _state_registry[h] = state
    return h

def state_from_hash(h):
    return _state_registry[h]


# ---------- Players ----------
def random_player(game, state):
    return random.choice(list(game.actions(state)))

def player(search_algorithm):
    """
    Wrap a search function that returns (value, move) or that returns a move directly.
    Returns a function (game, state) -> move to match play_game interface.
    """
    def chooser(game, state):
        out = search_algorithm(game, state)
        # If returns pair (value, move), return move; otherwise assume it's a move
        if isinstance(out, tuple) and len(out) == 2:
            return out[1]
        return out
    return chooser


# ---------- Game classes ----------
class Game:
    """Base class for a turn-taking game."""
    def actions(self, state):
        raise NotImplementedError
    def result(self, state, move):
        raise NotImplementedError
    def is_terminal(self, state):
        return not self.actions(state)
    def utility(self, state, player):
        raise NotImplementedError


def play_game(game, strategies: dict, verbose=False):
    """
    Play a turn-taking game. `strategies` is a dict {player_name: function(game, state) -> move}.
    Returns the terminal state.
    """
    state = game.initial
    while not game.is_terminal(state):
        player_to_move = state.to_move
        move = strategies[player_to_move](game, state)
        state = game.result(state, move)
        if verbose:
            print('Player', player_to_move, 'move:', move)
            print(state)
    return state


def k_in_row(board, player, square, k):
    """True if player has k pieces in a line through square."""
    def count_in_row(x, y, dx, dy):
        count = 0
        while (x, y) in board and board[(x, y)] == player:
            count += 1
            x, y = x + dx, y + dy
        return count

    return any(
        count_in_row(*square, dx, dy) + count_in_row(*square, -dx, -dy) - 1 >= k
        for (dx, dy) in ((0, 1), (1, 0), (1, 1), (1, -1))
    )


class Board(dict):
    """
    Board implements mapping (x,y) -> 'X'/'O' for occupied squares.
    It returns board.empty for in-range empty cells, board.off for out-of-range coords.
    Also stores width, height, to_move, utility (cached).
    """
    empty = '.'
    off = '#'

    def __init__(self, width=8, height=8, to_move=None, **kwds):
        dict.__init__(self)
        self.width = width
        self.height = height
        self.to_move = to_move
        self.utility = kwds.get('utility', 0)
        # optional: store k (winning length) on board if desired
        if 'k' in kwds:
            self.k = kwds['k']

    def new(self, changes: dict, **kwds) -> 'Board':
        "Return a new Board with the changes applied (functional style)."
        board = Board(width=self.width, height=self.height, to_move=kwds.get('to_move', self.to_move))
        board.update(self)   # copy entries
        board.update(changes)
        board.utility = kwds.get('utility', getattr(self, 'utility', 0))
        if hasattr(self, 'k'):
            board.k = self.k
        return board

    def __missing__(self, loc):
        x, y = loc
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.empty
        else:
            return self.off

    def __hash__(self):
        # Hash based on sorted occupied items + to_move + utility
        return hash((tuple(sorted(self.items())), self.to_move, self.utility))

    def __repr__(self):
        def row(y): return ' '.join(self[(x, y)] for x in range(self.width))
        return '\n'.join(map(row, range(self.height))) +  '\n'


class TicTacToe(Game):
    """A general TicTacToe on width x height needing k in a row to win."""
    def __init__(self, height=5, width=5, k=4):
        self.k = k
        self.width = width
        self.height = height
        # set of all coordinates
        self.squares = {(x, y) for x in range(width) for y in range(height)}
        # initial board also knows k for heuristic convenience
        self.initial = Board(width=width, height=height, to_move='X', utility=0, k=k)

    def actions(self, board):
        """Legal moves are any square not yet taken. Return as set of (x,y)."""
        return self.squares - set(board.keys())

    def result(self, board, square):
        """Place a marker for current player on square and return new Board."""
        player = board.to_move
        new_board = board.new({square: player}, to_move=('O' if player == 'X' else 'X'))
        # determine win on this placement
        win = k_in_row(new_board, player, square, self.k)
        new_board.utility = (0 if not win else +1 if player == 'X' else -1)
        # carry k into board for heuristic
        new_board.k = self.k
        return new_board

    def utility(self, board, player):
        return board.utility if player == 'X' else -board.utility

    def is_terminal(self, board):
        return board.utility != 0 or len(self.squares) == len(board)

    def display(self, board):
        print(board)


# ---------- __main__ example ----------
if __name__ == '__main__':
    # Create a 5x5 TicTacToe game with k=4 (four-in-a-row on 5x5)
    game = TicTacToe(height=5, width=5, k=4)

    # Create strategies:
    # X: random
    # O: alpha-beta with heuristic & cutoff depth 4
    cutoff = cutoff_depth(4)
    heur = heuristic_5x5_tictactoe

    # strategies_playoff = {
    #     'X': random_player,
    #     'O': player(lambda g, s: h_alphabeta_search(g, s, cutoff=cutoff, h=heur))
    # }


    strategies_playoff = {
        'X': minimax_player,
        'O': alphabeta_player(cutoff=cutoff, h=heur)
    }


    final_state = play_game(game, strategies_playoff, verbose=True)
    print("Final utility (from X's perspective):", final_state.utility)
    print("Final board:")
    print(final_state)