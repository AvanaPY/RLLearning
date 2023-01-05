from environments.chess.team import Team

def get_pawn_direction(team : Team):
    return {
        Team.WHITE : 1,
        Team.BLACK : -1
    }[team]