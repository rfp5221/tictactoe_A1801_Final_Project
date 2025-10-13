import math
import functools
infinity = math.inf
cache = functools.lru_cache(10**6)
def cache1(function):
    "Like lru_cache(None), but only considers the first argument of function."
    cache = {}
    def wrapped(x, *args):
        if x not in cache:
            cache[x] = function(x, *args)
        return cache[x]
    return wrapped

def alphabeta_search_tt(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
    def min_value(state, alpha, beta):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a), alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha:
                return v, move
        return v, move

    return max_value(state, -infinity, +infinity)

def minimax_search_tt(game, state):
    """Search game to determine best move; return (value, move) pair."""

    player = state.to_move

    @cache
    def max_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a))
            if v2 > v:
                v, move = v2, a
        return v, move

    @cache
    def min_value(state):
        if game.is_terminal(state):
            return game.utility(state, player), None
        v, move = +infinity, None
        for a in game.actions(state):
            v2, _ = max_value(game.result(state, a))
            if v2 < v:
                v, move = v2, a
        return v, move

    return max_value(state)

def cutoff_depth(d):
    """A cutoff function that searches to depth d."""
    return lambda game, state, depth: depth > d

def h_alphabeta_search(game, state, cutoff=cutoff_depth(6), h=lambda s, p: 0):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = state.to_move

    @cache1
    def max_value(state, alpha, beta, depth):
        if game.is_terminal(state):
            return game.utility(state, player), None
        if cutoff(game, state, depth):
            return h(state, player), None
        v, move = -infinity, None
        for a in game.actions(state):
            v2, _ = min_value(game.result(state, a), alpha, beta, depth+1)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta:
                return v, move
        return v, move

    @cache1
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
    """Heuristic evaluation function for a 5x5 Tic Tac Toe game."""
    def line_score(line):
        """Evaluate a line (row, column, or diagonal) for the player."""
        score = 0
        if line.count(player) > 0 and line.count(None) + line.count(player) == len(line):
            score += line.count(player) ** 2
        if line.count(opponent) > 0 and line.count(None) + line.count(opponent) == len(line):
            score -= line.count(opponent) ** 2
        return score

    board = state.board
    opponent = 'X' if player == 'O' else 'O'
    size = 5
    total_score = 0

    # Evaluate rows and columns
    for i in range(size):
        total_score += line_score([board.get((i, j)) for j in range(size)])  # Row
        total_score += line_score([board.get((j, i)) for j in range(size)])  # Column

    # Evaluate diagonals
    total_score += line_score([board.get((i, i)) for i in range(size)])  # Main diagonal
    total_score += line_score([board.get((i, size - 1 - i)) for i in range(size)])  # Anti-diagonal

    return total_score

# Example usage:
# h_alphabeta_search(game, state, cutoff=cutoff_depth(6), h=heuristic_5x5_tictactoe)