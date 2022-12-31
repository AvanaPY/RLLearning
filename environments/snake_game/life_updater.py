from abc import abstractclassmethod

try:
    from environments.snake_game.enums import ActionResult
    from environments.snake_game.context import GameContext
except ModuleNotFoundError:
    from enums import ActionResult
    from context import GameContext
class BaseLifeUpdater:
    def __init__(self):
        pass

    @abstractclassmethod
    def update_life(self, current_life : int, game_context : GameContext):
        pass

class ResetWhenAppleEatenLifeUpdater(BaseLifeUpdater):
    def __init__(self, reset_value : int):
        self._reset_value = reset_value

    def update_life(self, current_life : int, game_context : GameContext):
        if game_context.action_result == ActionResult.ATE_FOOD:
            return self._reset_value
        return current_life

class AdditiveWhenAppleEatenLifeUpdater(BaseLifeUpdater):
    def __init__(self, additive_value : int):
        self._additive_value = additive_value

    def update_life(self, current_value : int, game_context : GameContext):
        if game_context.action_result == ActionResult.ATE_FOOD:
            return current_value + self._additive_value
        return current_value

class ScoreMultiplyWhenAppleEatenLifeUpdater(BaseLifeUpdater):
    def __init__(self, factor : int):
        self._factor = factor

    def update_life(self, current_value : int, game_context : GameContext):
        if game_context.action_result != ActionResult.ATE_FOOD:
            return current_value
        return current_value + game_context.score * self._factor

class InfiniteLifeUpdater(BaseLifeUpdater):
    def __init__(self):
        pass

    def update_life(self, current_value : int, game_context : GameContext):
        return 9999