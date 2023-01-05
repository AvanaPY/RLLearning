from environments.chess.chess import ChessGame, Position
from environments.chess.piece import Piece
from environments.chess.move_generator import MoveGenerator

game = ChessGame()
mg = MoveGenerator(game)

game.move_piece_to_position(Position(0, 6), Position(0, 2))
game.move_piece_to_position(Position(1, 0), Position(2, 2))
game.move_piece_to_position(Position(4, 0), Position(3, 3))
moves = mg.generate_moves_for_position(Position(3, 3))
print(moves)
game.print_board()