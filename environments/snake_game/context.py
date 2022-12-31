try:
    from environments.snake_game.enums import ActionResult
except ModuleNotFoundError:
    from enums import ActionResult
    
class GameContext:
    def __init__(self, score : int, action_result : ActionResult):
        self._score = score
        self._ar = action_result

    @property
    def score(self):
        return self._score

    @property
    def action_result(self):
        return self._ar

    @staticmethod
    def GameOver(self, score : int):
        return GameContext(score, ActionResult.GAME_OVER)