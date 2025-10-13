import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.Game import play_game
from util.TicTacToe import TicTacToe
from util.Player import random_player, player
from util import Heuristics as H


play_game(TicTacToe(), dict(X=(random_player), O=player(H.minimax_search_tt)), verbose=True).utility
# # play_game(TicTacToe(), {'X':player(H.alphabeta_search_tt), 'O':player(H.minimax_search_tt)})
# play_game(TicTacToe(), {'X':player(H.heuristic_5x5_tictactoe), 'O':H.heuristic_5x5_tictactoe}, verbose=True)